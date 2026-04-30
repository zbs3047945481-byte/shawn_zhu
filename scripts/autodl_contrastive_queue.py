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
    for metrics_path in (repo_root / "result").rglob("*metrics.json"):
        if metrics_path.name not in {"metrics.json", "live_metrics.json"}:
            continue
        if prefix not in str(metrics_path.parent):
            continue
        metrics = read_json(metrics_path)
        options = metrics.get("options", {})
        rows.append({
            "tag": options.get("experiment_tag", metrics_path.parent.name),
            "plugin": metrics.get("plugin_name", ""),
            "k": options.get("fedfed_num_prototypes_per_class", ""),
            "momentum": options.get("fedfed_prototype_momentum", ""),
            "lambda_distill": options.get("fedfed_lambda_distill", ""),
            "lambda_contrastive": options.get("fedfed_lambda_contrastive", ""),
            "warmup": options.get("fedfed_distill_warmup_rounds", ""),
            "batch_size": options.get("batch_size", ""),
            "workers": options.get("dataloader_num_workers", ""),
            "round": metrics.get("final_round", ""),
            "final_acc": metrics.get("final_test_acc", ""),
            "best_acc": metrics.get("best_test_acc", ""),
            "metric_path": str(metrics_path),
        })
    rows.sort(key=lambda row: row["tag"])
    output_dir = repo_root / "outputs" / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tag", "plugin", "k", "momentum", "lambda_distill", "lambda_contrastive",
        "warmup", "batch_size", "workers", "round", "final_acc", "best_acc", "metric_path",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "summary.json", rows)
    return rows


def build_args(prefix):
    run_id = f"{prefix}_k2_ema0_lam03_sep005_wu5_a03_ep5"
    return run_id, [
        "--round_num", "100",
        "--num_of_clients", "20",
        "--c_fraction", "0.2",
        "--local_epoch", "5",
        "--batch_size", "256",
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
        "--fedfed_num_prototypes_per_class", "2",
        "--fedfed_min_samples_per_prototype", "8",
        "--fedfed_prototype_kmeans_iters", "8",
        "--fedfed_prototype_momentum", "0",
        "--fedfed_enable_proto_cls", "false",
        "--fedfed_enable_clip", "false",
        "--fedfed_enable_noise", "false",
        "--fedfed_adaptive_control", "false",
        "--fedfed_enable_anchor", "true",
        "--fedfed_lambda_anchor", "0.1",
        "--fedfed_enable_distill", "true",
        "--fedfed_lambda_distill", "0.3",
        "--fedfed_distill_warmup_rounds", "5",
        "--fedfed_enable_contrastive_distill", "true",
        "--fedfed_lambda_contrastive", "0.05",
        "--fedfed_contrastive_temperature", "0.2",
        "--experiment_tag", run_id,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runs_dir = repo_root / "runs" / args.prefix
    runs_dir.mkdir(parents=True, exist_ok=True)
    status_path = runs_dir / "status.json"
    run_id, train_args = build_args(args.prefix)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(status_path, {
        "prefix": args.prefix,
        "status": "running",
        "current_run": run_id,
        "started_at": utc_now(),
        "experiments_total": 1,
        "experiments_done": 0,
    })
    command = [sys.executable, "-u", "main.py"] + train_args
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout_file, \
            (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr_file:
        process = subprocess.run(command, cwd=repo_root, stdout=stdout_file, stderr=stderr_file)

    rows = collect_summary(repo_root, args.prefix)
    write_json(status_path, {
        "prefix": args.prefix,
        "status": "succeeded" if process.returncode == 0 else "failed",
        "current_run": run_id,
        "finished_at": utc_now(),
        "experiments_total": 1,
        "experiments_done": 1 if process.returncode == 0 else 0,
        "exit_code": process.returncode,
        "summary": rows,
    })
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
