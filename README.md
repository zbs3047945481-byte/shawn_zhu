# Plug-and-Play Feature Distillation for Federated Learning

This repository implements a lightweight, folder-level plugin for feature distillation in federated learning.

Current focus:

- `FedAvg`-style federated training baseline
- `fedfed_prototype` plugin with class-prototype distillation
- configurable Non-IID simulation
- experiment plotting and batch experiment suites

## Quick Start

Create environment:

```bash
conda env create -f environment.yml
conda activate t
```

Run a minimal training example:

```bash
python main.py \
  --round_num 1 \
  --num_of_clients 5 \
  --c_fraction 0.4 \
  --local_epoch 1 \
  --batch_size 64 \
  --gpu false \
  --dataset_name mnist \
  --partition_strategy dirichlet \
  --dirichlet_alpha 0.3 \
  --enable_quantity_skew true \
  --enable_feature_skew true \
  --plugin_name fedfed_prototype
```

## Outputs

Each experiment writes results under `result/<dataset>/<experiment>/`, including:

- `metrics.json`
- `test_acc_curve.png`
- `test_loss_curve.png`

## Common Commands

Compare multiple finished runs:

```bash
python plot_experiments.py \
  --metrics path/to/exp1/metrics.json path/to/exp2/metrics.json \
  --labels FedAvg FedFed \
  --output_dir result/comparisons
```

Run a predefined experiment suite:

```bash
python run_experiment_suite.py --suite thesis_main
```

## Recommended Reading Order

Core usage and integration:

- [Plugin Overview](docs/FEDFED_PLUGIN.md)
- [External Integration Guide](docs/EXTERNAL_INTEGRATION_GUIDE.md)
- [Experiment Visualization](docs/EXPERIMENT_VISUALIZATION.md)
- [Thesis Experiment Plan](docs/THESIS_EXPERIMENT_PLAN.md)
- [Thesis Execution Checklist](docs/THESIS_EXECUTION_CHECKLIST.md)

Development history:

- [Docs Index](docs/README.md)

## Project Structure

Main entry and orchestration:

- `main.py`
- `src/fed_server/`
- `src/fed_client/`

Plugin implementation:

- `src/plugins/fedfed_plugin.py`
- `src/plugins/base.py`
- `src/plugins/minimal_template.py`

Model and feature module:

- `src/models/mnist_cnn.py`
- `src/models/feature_split.py`

Experiment tooling:

- `plot_experiments.py`
- `run_experiment_suite.py`
- `src/utils/metrics.py`
- `src/utils/plotting.py`
