#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_summary(repo_root, prefix):
    rows = []
    result_root = repo_root / "result"
    if result_root.exists():
        grouped = {}
        for metrics_path in result_root.rglob("*metrics.json"):
            if metrics_path.name in {"metrics.json", "live_metrics.json"} and prefix in str(metrics_path.parent):
                grouped.setdefault(metrics_path.parent, []).append(metrics_path)
        for exp_dir, files in grouped.items():
            metrics_path = sorted(files, key=lambda p: 0 if p.name == "metrics.json" else 1)[0]
            metrics = read_json(metrics_path)
            options = metrics.get("options", {})
            rows.append({
                "tag": options.get("experiment_tag", exp_dir.name),
                "plugin": metrics.get("plugin_name", ""),
                "k": options.get("fedfed_num_prototypes_per_class", ""),
                "momentum": options.get("fedfed_prototype_momentum", ""),
                "lambda_distill": options.get("fedfed_lambda_distill", ""),
                "warmup": options.get("fedfed_distill_warmup_rounds", ""),
                "warmup_mode": options.get("fedfed_distill_warmup_mode", "linear"),
                "batch_size": options.get("batch_size", ""),
                "workers": options.get("dataloader_num_workers", ""),
                "round": metrics.get("final_round", ""),
                "final_acc": metrics.get("final_test_acc", ""),
                "best_acc": metrics.get("best_test_acc", ""),
                "metric_file": metrics_path.name,
                "metric_path": str(metrics_path),
            })
    rows.sort(key=lambda row: row["tag"])
    output_dir = repo_root / "outputs" / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tag", "plugin", "k", "momentum", "lambda_distill", "warmup", "warmup_mode",
        "batch_size", "workers", "round", "final_acc", "best_acc", "metric_file", "metric_path",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "summary.json", rows)
    return rows


def build_experiments(prefix, round_num, batch_size, workers):
    base_common = [
        "--round_num", str(round_num),
        "--num_of_clients", "20",
        "--c_fraction", "0.2",
        "--local_epoch", "5",
        "--batch_size", str(batch_size),
        "--dataloader_num_workers", str(workers),
        "--dataloader_pin_memory", "true",
        "--torch_cudnn_benchmark", "true",
        "--gpu", "true",
        "--dataset_name", "cifar10",
        "--partition_strategy", "dirichlet",
        "--dirichlet_alpha", "0.3",
        "--min_samples_per_client", "1",
        "--enable_quantity_skew", "true",
        "--enable_feature_skew", "true",
        "--lr", "0.001",
        "--early_stop_enable", "true",
        "--early_stop_min_rounds", "40",
        "--early_stop_patience", "20",
        "--early_stop_min_delta", "0.002",
    ]
    plugin_common = base_common + [
        "--plugin_name", "fedfed_prototype",
        "--fedfed_enable_projection", "false",
        "--fedfed_sensitive_dim", "512",
        "--fedfed_distill_count_tau", "0",
        "--fedfed_normalize_prototypes", "false",
        "--fedfed_use_cosine_distill", "false",
        "--fedfed_prototype_source", "reference",
        "--fedfed_reference_proto_max_batches", "0",
        "--fedfed_min_samples_per_prototype", "8",
        "--fedfed_prototype_kmeans_iters", "8",
        "--fedfed_enable_proto_cls", "false",
        "--fedfed_enable_clip", "false",
        "--fedfed_enable_noise", "false",
        "--fedfed_adaptive_control", "false",
        "--fedfed_enable_anchor", "true",
        "--fedfed_lambda_anchor", "0.1",
        "--fedfed_enable_distill", "true",
        "--fedfed_distill_warmup_rounds", "5",
    ]
    fedavg_id = f"{prefix}_fedavg_a03_ep5"
    experiments = [{
        "run_id": fedavg_id,
        "args": base_common + [
            "--plugin_name", "none",
            "--experiment_tag", fedavg_id,
        ],
    }]
    configs = [
        ("single_k1_ema0_lam03_wu5", 1, 0.0, 0.3),
        ("multi_k2_ema0_lam03_wu5", 2, 0.0, 0.3),
        ("multi_k2_ema0_lam05_wu5", 2, 0.0, 0.5),
        ("multi_k2_ema03_lam03_wu5", 2, 0.3, 0.3),
    ]
    for name, k_value, momentum, lambda_distill in configs:
        run_id = f"{prefix}_{name}_a03_ep5"
        experiments.append({
            "run_id": run_id,
            "args": plugin_common + [
                "--fedfed_num_prototypes_per_class", str(k_value),
                "--fedfed_prototype_momentum", str(momentum),
                "--fedfed_lambda_distill", str(lambda_distill),
                "--experiment_tag", run_id,
            ],
        })
    return experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--round-num", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = repo_root / "runs" / args.prefix
    status_path = runs_dir / "status.json"
    runs_dir.mkdir(parents=True, exist_ok=True)
    experiments = build_experiments(args.prefix, args.round_num, args.batch_size, args.workers)

    status = {
        "prefix": args.prefix,
        "status": "running",
        "current_run": "",
        "started_at": utc_now(),
        "finished_at": None,
        "experiments_total": len(experiments),
        "experiments_done": 0,
    }
    write_json(status_path, status)

    for index, experiment in enumerate(experiments, start=1):
        run_id = experiment["run_id"]
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run_status_path = run_dir / "run_status.json"
        status.update({"status": "running", "current_run": run_id, "experiments_done": index - 1, "updated_at": utc_now()})
        write_json(status_path, status)
        write_json(run_status_path, {"run_id": run_id, "status": "running", "started_at": utc_now(), "exit_code": None})

        command = [sys.executable, "-u", "main.py"] + experiment["args"]
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout_file, \
                (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr_file:
            process = subprocess.run(command, cwd=repo_root, stdout=stdout_file, stderr=stderr_file)

        write_json(run_status_path, {
            "run_id": run_id,
            "status": "succeeded" if process.returncode == 0 else "failed",
            "started_at": read_json(run_status_path)["started_at"],
            "finished_at": utc_now(),
            "exit_code": process.returncode,
        })
        collect_summary(repo_root, args.prefix)
        if process.returncode != 0:
            status.update({"status": "failed", "current_run": run_id, "finished_at": utc_now(), "exit_code": process.returncode})
            write_json(status_path, status)
            return process.returncode

    collect_summary(repo_root, args.prefix)
    status.update({
        "status": "succeeded",
        "current_run": "",
        "finished_at": utc_now(),
        "experiments_done": len(experiments),
    })
    write_json(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
