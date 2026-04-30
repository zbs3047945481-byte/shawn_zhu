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


def find_metric_files(result_root, prefix):
    if not result_root.exists():
        return []
    return [
        path for path in result_root.rglob("*metrics.json")
        if prefix in str(path.parent) and path.name in {"metrics.json", "live_metrics.json"}
    ]


def write_summary(repo_root, prefix):
    grouped = {}
    for metrics_path in find_metric_files(repo_root / "result", prefix):
        grouped.setdefault(metrics_path.parent, []).append(metrics_path)

    rows = []
    for exp_dir, files in grouped.items():
        metrics_path = sorted(files, key=lambda p: 0 if p.name == "metrics.json" else 1)[0]
        metrics = read_json(metrics_path)
        options = metrics.get("options", {})
        tag = options.get("experiment_tag", exp_dir.name)
        rows.append({
            "tag": tag,
            "method": tag.replace(prefix + "_", ""),
            "plugin": metrics.get("plugin_name", ""),
            "alpha": options.get("dirichlet_alpha", ""),
            "epoch": options.get("local_epoch", ""),
            "round": metrics.get("final_round", ""),
            "final_acc": metrics.get("final_test_acc", ""),
            "best_acc": metrics.get("best_test_acc", ""),
            "metric_file": metrics_path.name,
            "metric_path": str(metrics_path),
        })

    rows.sort(key=lambda row: row["tag"])
    output_dir = repo_root / "outputs" / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
            "tag", "method", "plugin", "alpha", "epoch", "round", "final_acc", "best_acc", "metric_file", "metric_path"
        ])
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "summary.json", rows)
    return rows


def build_experiments(prefix, round_num, batch_size):
    common = [
        "--round_num", str(round_num),
        "--num_of_clients", "20",
        "--c_fraction", "0.2",
        "--local_epoch", "5",
        "--batch_size", str(batch_size),
        "--dataloader_num_workers", "0",
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
        "--plugin_name", "fedfed_prototype",
        "--fedfed_enable_projection", "false",
        "--fedfed_sensitive_dim", "512",
        "--fedfed_distill_count_tau", "0",
        "--fedfed_normalize_prototypes", "false",
        "--fedfed_use_cosine_distill", "false",
        "--fedfed_prototype_source", "reference",
        "--fedfed_reference_proto_max_batches", "0",
        "--fedfed_enable_proto_cls", "false",
        "--fedfed_enable_clip", "false",
        "--fedfed_enable_noise", "false",
        "--fedfed_adaptive_control", "false",
        "--fedfed_enable_anchor", "true",
        "--fedfed_lambda_anchor", "0.1",
        "--fedfed_enable_distill", "true",
        "--fedfed_lambda_distill", "0.3",
        "--fedfed_distill_warmup_rounds", "5",
        "--fedfed_distill_warmup_mode", "hard",
    ]
    experiments = []
    for momentum in ("0.3", "0.5"):
        run_id = f"{prefix}_lambda03_hardwarmup5_ema{momentum.replace('.', '')}_a03_ep5"
        experiments.append({
            "run_id": run_id,
            "args": common + [
                "--fedfed_prototype_momentum", momentum,
                "--experiment_tag", run_id,
            ],
        })
    return experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="autodl_hard_warmup_ema")
    parser.add_argument("--round-num", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = repo_root / "runs" / args.prefix
    status_path = runs_dir / "status.json"
    runs_dir.mkdir(parents=True, exist_ok=True)
    experiments = build_experiments(args.prefix, args.round_num, args.batch_size)

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
        status.update({"current_run": run_id, "experiments_done": index - 1, "updated_at": utc_now()})
        write_json(status_path, status)
        write_json(run_status_path, {"run_id": run_id, "status": "running", "started_at": utc_now()})

        command = [sys.executable, "-u", "main.py"] + experiment["args"]
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout_file, \
                (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr_file:
            process = subprocess.run(command, cwd=repo_root, stdout=stdout_file, stderr=stderr_file)

        write_json(run_status_path, {
            "run_id": run_id,
            "status": "succeeded" if process.returncode == 0 else "failed",
            "finished_at": utc_now(),
            "exit_code": process.returncode,
        })
        write_summary(repo_root, args.prefix)
        if process.returncode != 0:
            status.update({"status": "failed", "current_run": run_id, "finished_at": utc_now()})
            write_json(status_path, status)
            return process.returncode

    write_summary(repo_root, args.prefix)
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
