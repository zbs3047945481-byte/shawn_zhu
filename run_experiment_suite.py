import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

from plot_experiments import load_experiments
from src.options import str2bool
from src.utils.plotting import plotting_available, save_ablation_summary_plot, save_comparison_plots


SUITES = {
    'baseline_vs_plugin': [
        {'label': 'FedAvg', 'args': ['--plugin_name', 'none']},
        {'label': 'FedFedPrototype', 'args': ['--plugin_name', 'fedfed_prototype']},
    ],
    'fedavg_alpha_main': [
        {'label': 'FedAvg_alpha_1.0', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '1.0']},
        {'label': 'FedAvg_alpha_0.5', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.5']},
        {'label': 'FedAvg_alpha_0.3', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.3']},
        {'label': 'FedAvg_alpha_0.1', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.1']},
    ],
    'fedavg_stress': [
        {'label': 'alpha_0.3_ep5', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.3', '--local_epoch', '5']},
        {'label': 'alpha_0.1_ep5', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.1', '--local_epoch', '5']},
        {'label': 'alpha_0.05_ep5', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.05', '--local_epoch', '5']},
        {'label': 'alpha_0.01_ep5', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.01', '--local_epoch', '5']},
        {'label': 'alpha_0.3_ep10', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.3', '--local_epoch', '10']},
        {'label': 'alpha_0.1_ep10', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.1', '--local_epoch', '10']},
        {'label': 'alpha_0.05_ep10', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.05', '--local_epoch', '10']},
        {'label': 'alpha_0.01_ep10', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.01', '--local_epoch', '10']},
    ],
    'fedavg_collapse_probe': [
        {'label': 'alpha_0.1_ep10_min32', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.1', '--local_epoch', '10', '--min_samples_per_client', '32']},
        {'label': 'alpha_0.05_ep10_min16', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.05', '--local_epoch', '10', '--min_samples_per_client', '16']},
        {'label': 'alpha_0.01_ep10_min8', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.01', '--local_epoch', '10', '--min_samples_per_client', '8']},
        {'label': 'alpha_0.01_ep20_min8', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.01', '--local_epoch', '20', '--min_samples_per_client', '8']},
        {'label': 'alpha_0.005_ep20_min4', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.005', '--local_epoch', '20', '--min_samples_per_client', '4']},
    ],
    'fedavg_vs_full_grid': [
        {'label': 'FedAvg_alpha_0.5_ep5', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.5', '--local_epoch', '5']},
        {'label': 'Full_alpha_0.5_ep5', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '0.5', '--local_epoch', '5']},
        {'label': 'FedAvg_alpha_0.3_ep5', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.3', '--local_epoch', '5']},
        {'label': 'Full_alpha_0.3_ep5', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '0.3', '--local_epoch', '5']},
        {'label': 'FedAvg_alpha_0.1_ep5', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.1', '--local_epoch', '5']},
        {'label': 'Full_alpha_0.1_ep5', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '0.1', '--local_epoch', '5']},
        {'label': 'FedAvg_alpha_0.5_ep10', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.5', '--local_epoch', '10']},
        {'label': 'Full_alpha_0.5_ep10', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '0.5', '--local_epoch', '10']},
        {'label': 'FedAvg_alpha_0.3_ep10', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.3', '--local_epoch', '10']},
        {'label': 'Full_alpha_0.3_ep10', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '0.3', '--local_epoch', '10']},
        {'label': 'FedAvg_alpha_0.1_ep10', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.1', '--local_epoch', '10']},
        {'label': 'Full_alpha_0.1_ep10', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '0.1', '--local_epoch', '10']},
    ],
    'targeted_four_combo': [
        {'label': 'FedAvg_alpha_0.3_ep1', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.3', '--local_epoch', '1']},
        {'label': 'Full_alpha_0.3_ep1', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '0.3', '--local_epoch', '1']},
        {'label': 'FedAvg_alpha_0.1_ep1', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.1', '--local_epoch', '1']},
        {'label': 'Full_alpha_0.1_ep1', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '0.1', '--local_epoch', '1']},
        {'label': 'FedAvg_alpha_0.3_ep5', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.3', '--local_epoch', '5']},
        {'label': 'Full_alpha_0.3_ep5', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '0.3', '--local_epoch', '5']},
        {'label': 'FedAvg_alpha_0.1_ep5', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.1', '--local_epoch', '5']},
        {'label': 'Full_alpha_0.1_ep5', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '0.1', '--local_epoch', '5']},
    ],
    'alpha_sweep': [
        {'label': 'alpha_1.0', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '1.0']},
        {'label': 'alpha_0.5', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '0.5']},
        {'label': 'alpha_0.3', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '0.3']},
        {'label': 'alpha_0.1', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '0.1']},
    ],
    'lambda_sweep': [
        {'label': 'lambda_0.1', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_lambda_distill', '0.1']},
        {'label': 'lambda_0.5', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_lambda_distill', '0.5']},
        {'label': 'lambda_1.0', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_lambda_distill', '1.0']},
        {'label': 'lambda_2.0', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_lambda_distill', '2.0']},
    ],
    'dim_sweep': [
        {'label': 'dim_16', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_sensitive_dim', '16']},
        {'label': 'dim_32', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_sensitive_dim', '32']},
        {'label': 'dim_64', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_sensitive_dim', '64']},
        {'label': 'dim_128', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_sensitive_dim', '128']},
    ],
    'thesis_main': [
        {'label': 'FedAvg_alpha_1.0', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '1.0']},
        {'label': 'FedFed_alpha_1.0', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '1.0']},
        {'label': 'FedAvg_alpha_0.5', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.5']},
        {'label': 'FedFed_alpha_0.5', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '0.5']},
        {'label': 'FedAvg_alpha_0.3', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.3']},
        {'label': 'FedFed_alpha_0.3', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '0.3']},
        {'label': 'FedAvg_alpha_0.1', 'args': ['--plugin_name', 'none', '--dirichlet_alpha', '0.1']},
        {'label': 'FedFed_alpha_0.1', 'args': ['--plugin_name', 'fedfed_prototype', '--dirichlet_alpha', '0.1']},
    ],
    'thesis_ablation': [
        {'label': 'FedAvg', 'args': ['--plugin_name', 'none']},
        {'label': 'Prototype_dim_64', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_sensitive_dim', '64']},
        {'label': 'Prototype_dim_64_no_noise', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_sensitive_dim', '64', '--fedfed_noise_sigma', '0.0']},
        {'label': 'Prototype_dim_64_low_lambda', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_sensitive_dim', '64', '--fedfed_lambda_distill', '0.1']},
    ],
    'thesis_heterogeneity': [
        {'label': 'label_only', 'args': ['--plugin_name', 'fedfed_prototype', '--enable_quantity_skew', 'false', '--enable_feature_skew', 'false', '--dirichlet_alpha', '0.3']},
        {'label': 'label_quantity', 'args': ['--plugin_name', 'fedfed_prototype', '--enable_quantity_skew', 'true', '--enable_feature_skew', 'false', '--dirichlet_alpha', '0.3']},
        {'label': 'label_feature', 'args': ['--plugin_name', 'fedfed_prototype', '--enable_quantity_skew', 'false', '--enable_feature_skew', 'true', '--dirichlet_alpha', '0.3']},
        {'label': 'label_quantity_feature', 'args': ['--plugin_name', 'fedfed_prototype', '--enable_quantity_skew', 'true', '--enable_feature_skew', 'true', '--dirichlet_alpha', '0.3']},
    ],
    'thesis_engineering': [
        {'label': 'FedAvg', 'args': ['--plugin_name', 'none']},
        {'label': 'FolderPlugin_default', 'args': ['--plugin_name', 'fedfed_prototype']},
        {'label': 'FolderPlugin_no_noise', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_noise_sigma', '0.0']},
    ],
    'midterm_ablation_core': [
        {'label': 'FedAvg', 'args': ['--plugin_name', 'none']},
        {'label': 'Full_Method', 'args': ['--plugin_name', 'fedfed_prototype']},
        {'label': 'No_Prototype_Sharing', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_enable_prototype_sharing', 'false']},
        {'label': 'No_Distill', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_enable_distill', 'false']},
        {'label': 'No_Projection', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_enable_projection', 'false']},
        {'label': 'No_Clip', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_enable_clip', 'false']},
    ],
    'midterm_distill_focus': [
        {'label': 'FedAvg', 'args': ['--plugin_name', 'none']},
        {'label': 'Full_Method', 'args': ['--plugin_name', 'fedfed_prototype']},
        {'label': 'No_Prototype_Sharing', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_enable_prototype_sharing', 'false']},
        {'label': 'No_Distill', 'args': ['--plugin_name', 'fedfed_prototype', '--fedfed_enable_distill', 'false']},
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run batch FL experiment suites and generate comparison plots.')
    parser.add_argument('--suite', choices=sorted(SUITES.keys()), required=True, help='Predefined experiment suite to run.')
    parser.add_argument('--round_num', type=int, default=5, help='Rounds for each experiment.')
    parser.add_argument('--num_of_clients', type=int, default=10, help='Total clients.')
    parser.add_argument('--c_fraction', type=float, default=0.2, help='Fraction of clients per round.')
    parser.add_argument('--local_epoch', type=int, default=1, help='Local epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Local batch size.')
    parser.add_argument('--gpu', type=str2bool, default=True, help='Whether to use GPU.')
    parser.add_argument('--dataset_name', type=str, default='mnist', help='Dataset name.')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3, help='Dirichlet alpha used in shared base settings.')
    parser.add_argument('--seed', type=int, default=3001, help='Base random seed.')
    parser.add_argument('--num_repeats', type=int, default=1, help='How many random-seed repeats to run for each configuration.')
    parser.add_argument('--output_root', type=str, default='result/suites', help='Output folder for suite-level comparisons.')
    parser.add_argument('--dry_run', action='store_true', help='Print commands without executing them.')
    return parser.parse_args()


def build_base_args(args):
    return [
        '--round_num', str(args.round_num),
        '--num_of_clients', str(args.num_of_clients),
        '--c_fraction', str(args.c_fraction),
        '--local_epoch', str(args.local_epoch),
        '--batch_size', str(args.batch_size),
        '--gpu', str(args.gpu).lower(),
        '--dataset_name', args.dataset_name,
        '--partition_strategy', 'dirichlet',
        '--dirichlet_alpha', str(args.dirichlet_alpha),
        '--enable_quantity_skew', 'true',
        '--enable_feature_skew', 'true',
    ]


def run_suite(args):
    suite_runs = SUITES[args.suite]
    suite_output_dir = Path(args.output_root) / args.suite
    suite_output_dir.mkdir(parents=True, exist_ok=True)

    metrics_paths = []
    labels = []
    repeated_results = []
    for index, run_cfg in enumerate(suite_runs):
        run_metrics_paths = []
        for repeat_idx in range(args.num_repeats):
            run_seed = args.seed + repeat_idx
            tag = '{}_{}_rep{}'.format(args.suite, run_cfg['label'], repeat_idx + 1)
            command = [
                sys.executable,
                'main.py',
                '--seed', str(run_seed),
                '--experiment_tag', tag,
            ] + build_base_args(args) + run_cfg['args']

            print('Running:', ' '.join(command))
            if args.dry_run:
                continue

            subprocess.run(command, check=True)

            metrics_path = find_metrics_path(args.dataset_name, run_seed, args.round_num, args.batch_size, tag)
            run_metrics_paths.append(str(metrics_path))

        if args.dry_run:
            continue

        run_experiments = load_experiments(run_metrics_paths, None)
        summary = aggregate_repeated_results(run_cfg['label'], run_experiments)
        repeated_results.append(summary)
        metrics_paths.append(run_metrics_paths[0])
        labels.append(run_cfg['label'])

    if args.dry_run:
        return

    if not plotting_available():
        raise RuntimeError('matplotlib is required to generate suite comparison plots.')

    experiments = load_experiments(metrics_paths, labels)
    save_comparison_plots(experiments, str(suite_output_dir))
    save_suite_summary(args.suite, experiments, metrics_paths, suite_output_dir, repeated_results)
    if args.suite == 'midterm_ablation_six':
        save_ablation_summary_plot(
            repeated_results,
            str(suite_output_dir / 'midterm_ablation_summary.png'),
            title='Midterm Six-Way Ablation',
        )
    print('Saved suite outputs to {}'.format(suite_output_dir))


def aggregate_repeated_results(label, experiments):
    metrics_list = [experiment['metrics'] for experiment in experiments]
    best_acc_values = [metrics.get('best_test_acc', 0.0) for metrics in metrics_list]
    final_acc_values = [metrics.get('final_test_acc', 0.0) for metrics in metrics_list]
    best_loss_values = [metrics.get('best_test_loss', 0.0) for metrics in metrics_list]
    final_loss_values = [metrics.get('final_test_loss', 0.0) for metrics in metrics_list]

    return {
        'label': label,
        'plugin_name': metrics_list[0].get('plugin_name', 'none'),
        'num_repeats': len(metrics_list),
        'best_test_acc_mean': mean(best_acc_values),
        'best_test_acc_std': _safe_std(best_acc_values),
        'final_test_acc_mean': mean(final_acc_values),
        'final_test_acc_std': _safe_std(final_acc_values),
        'best_test_loss_mean': mean(best_loss_values),
        'best_test_loss_std': _safe_std(best_loss_values),
        'final_test_loss_mean': mean(final_loss_values),
        'final_test_loss_std': _safe_std(final_loss_values),
    }


def _safe_std(values):
    if len(values) <= 1:
        return 0.0
    return pstdev(values)


def save_suite_summary(suite_name, experiments, metrics_paths, output_dir, repeated_results):
    summary_path = output_dir / 'suite_summary.json'
    summary_table_path = output_dir / 'suite_summary.csv'
    repeat_summary_path = output_dir / 'suite_summary_multiseed.csv'

    payload = {
        'suite': suite_name,
        'metrics': metrics_paths,
        'labels': [experiment['label'] for experiment in experiments],
        'results': [
            {
                'label': experiment['label'],
                'best_test_acc': experiment['metrics'].get('best_test_acc', 0.0),
                'final_test_acc': experiment['metrics'].get('final_test_acc', 0.0),
                'best_test_loss': experiment['metrics'].get('best_test_loss', 0.0),
                'final_test_loss': experiment['metrics'].get('final_test_loss', 0.0),
                'plugin_name': experiment['metrics'].get('plugin_name', 'none'),
            }
            for experiment in experiments
        ],
        'multiseed_results': repeated_results,
    }

    with open(summary_path, 'w') as outfile:
        json.dump(payload, outfile, indent=2)

    with open(summary_table_path, 'w', newline='') as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=['label', 'plugin_name', 'best_test_acc', 'final_test_acc', 'best_test_loss', 'final_test_loss'],
        )
        writer.writeheader()
        for row in payload['results']:
            writer.writerow(row)

    with open(repeat_summary_path, 'w', newline='') as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=[
                'label',
                'plugin_name',
                'num_repeats',
                'best_test_acc_mean',
                'best_test_acc_std',
                'final_test_acc_mean',
                'final_test_acc_std',
                'best_test_loss_mean',
                'best_test_loss_std',
                'final_test_loss_mean',
                'final_test_loss_std',
            ],
        )
        writer.writeheader()
        for row in repeated_results:
            writer.writerow(row)


def find_metrics_path(dataset_name, seed, round_num, batch_size, tag):
    result_dir = Path('result') / dataset_name
    matches = sorted(result_dir.glob('*_sd{}_lr*_ne{}_bs{}_{}'.format(seed, round_num, batch_size, tag)))
    if not matches:
        raise FileNotFoundError('Could not find experiment folder for tag {}'.format(tag))
    return matches[-1] / 'metrics.json'


def main():
    args = parse_args()
    run_suite(args)


if __name__ == '__main__':
    main()
