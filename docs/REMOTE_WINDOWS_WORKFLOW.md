# Remote Windows Workflow

This repository includes four scripts for the "edit on Mac, run on Windows GPU" workflow:

- `scripts/deploy_to_windows.sh`
- `scripts/run_remote_train.sh`
- `scripts/check_remote_run.sh`
- `scripts/run_windows_experiment.sh`

## 1. Fill in remote config

Copy `scripts/remote_windows.env.example` to `scripts/remote_windows.env`, then set:

- `WIN_HOST_ALIAS`: SSH alias in `~/.ssh/config`
- `REMOTE_PROJECT_DIR`: Windows code directory
- `REMOTE_RUNS_DIR`: Windows log/checkpoint root
- `REMOTE_Codex_DIR`: Windows-side scratch directory used for uploaded archives and helper scripts
- `REMOTE_PYTHON`: full path to the Windows `python.exe`

## 2. Deploy the current working tree

```bash
./scripts/deploy_to_windows.sh
```

The deploy script is incremental. It computes a file manifest on macOS, compares it with the last deployed manifest on Windows, then:

- uploads only changed or newly added files
- deletes files on Windows that were removed locally
- leaves `data/`, `result/`, and `outputs/` untouched

This is much faster than full replacement when you only changed a few source files.

## 3. Start remote training

```bash
./scripts/run_remote_train.sh -- \
  --round_num 1 \
  --num_of_clients 5 \
  --c_fraction 0.4 \
  --local_epoch 1 \
  --batch_size 64 \
  --gpu true \
  --dataset_name mnist \
  --partition_strategy dirichlet \
  --dirichlet_alpha 0.3 \
  --enable_quantity_skew true \
  --enable_feature_skew true \
  --plugin_name fedfed_prototype
```

If you do not provide `--experiment_tag`, the script injects the generated `run_id` automatically.

## 4. Check logs and GPU status

```bash
./scripts/check_remote_run.sh
./scripts/check_remote_run.sh --run-id run_20260420T120000Z
```

The check script prints:

- selected run directory
- stored launch metadata
- durable run status when available
- current process state
- `nvidia-smi` GPU summary
- tail of `stdout.log` and `stderr.log`
- newest `metrics.json` under `result/`

## 5. One-command experiment workflow

For reliable day-to-day use, prefer the foreground wrapper instead of the background launcher:

```bash
./scripts/run_windows_experiment.sh --preset plugin-smoke
```

It performs:

- deploy current local code to Windows
- run `main.py` on Windows in the foreground
- stream training output back to this Mac
- write `stdout.log` under `D:\runs/<run_id>/`
- print the final remote status summary

You can also pass a custom experiment command:

```bash
./scripts/run_windows_experiment.sh --run-id thesis_alpha_03 -- \
  --round_num 20 \
  --num_of_clients 20 \
  --c_fraction 0.2 \
  --local_epoch 5 \
  --batch_size 64 \
  --gpu true \
  --dataset_name mnist \
  --partition_strategy dirichlet \
  --dirichlet_alpha 0.3 \
  --enable_quantity_skew true \
  --enable_feature_skew true \
  --plugin_name fedfed_prototype
```

Useful options:

- `--skip-deploy`: reuse the current Windows code copy
- `--preset plugin-smoke`: fast GPU smoke run with `fedfed_prototype`
- `--preset fedavg-smoke`: fast GPU smoke run with `plugin_name none`

## 6. Stable background runner

Use `scripts/run_remote_train.sh` for detached long runs:

```bash
./scripts/run_remote_train.sh --run-id thesis_long_run -- \
  --round_num 100 \
  --num_of_clients 20 \
  --c_fraction 0.2 \
  --local_epoch 5 \
  --batch_size 64 \
  --gpu true \
  --dataset_name mnist \
  --partition_strategy dirichlet \
  --dirichlet_alpha 0.3 \
  --enable_quantity_skew true \
  --enable_feature_skew true \
  --plugin_name fedfed_prototype
```

This command returns immediately after Windows launches a separate worker process. The detached worker keeps running after SSH exits and writes:

- `run_meta.json`: launch metadata
- `run_status.json`: `running`, `succeeded`, or `failed`, plus exit code
- `stdout.log`
- `stderr.log`
