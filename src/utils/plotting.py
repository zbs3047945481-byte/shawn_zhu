import os
import tempfile


try:
    os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'matplotlib-cache'))
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def plotting_available():
    return plt is not None


def save_single_run_plots(metrics, output_dir):
    if not plotting_available():
        return False

    rounds = metrics.get('rounds') or list(range(len(metrics['acc_on_g_test_data'])))
    acc_values = metrics['acc_on_g_test_data']
    loss_values = metrics['loss_on_g_test_data']

    _save_curve(
        rounds,
        acc_values,
        os.path.join(output_dir, 'test_acc_curve.png'),
        'Test Accuracy',
        'Round',
        'Accuracy',
    )
    _save_curve(
        rounds,
        loss_values,
        os.path.join(output_dir, 'test_loss_curve.png'),
        'Test Loss',
        'Round',
        'Loss',
    )
    _save_detailed_curve(
        rounds,
        acc_values,
        os.path.join(output_dir, 'test_acc_curve_detailed.png'),
        title='Detailed Test Accuracy by Round',
        ylabel='Accuracy',
        line_color='#2a6f97',
        fill_color='#cfe8f3',
    )
    _save_detailed_curve(
        rounds,
        loss_values,
        os.path.join(output_dir, 'test_loss_curve_detailed.png'),
        title='Detailed Test Loss by Round',
        ylabel='Loss',
        line_color='#c76d3a',
        fill_color='#f8dcc9',
    )
    return True


def save_comparison_plots(experiments, output_dir):
    if not plotting_available() or not experiments:
        return False

    os.makedirs(output_dir, exist_ok=True)
    _save_multi_curve(
        experiments,
        output_dir,
        metric_key='acc_on_g_test_data',
        filename='compare_test_acc.png',
        title='Test Accuracy Comparison',
        ylabel='Accuracy',
    )
    _save_multi_curve(
        experiments,
        output_dir,
        metric_key='loss_on_g_test_data',
        filename='compare_test_loss.png',
        title='Test Loss Comparison',
        ylabel='Loss',
    )
    _save_summary_bar(
        experiments,
        output_dir,
        metric_getter=lambda exp: exp['metrics'].get('best_test_acc', 0.0),
        filename='compare_best_acc_bar.png',
        title='Best Test Accuracy',
        ylabel='Accuracy',
    )
    _save_summary_bar(
        experiments,
        output_dir,
        metric_getter=lambda exp: exp['metrics'].get('final_test_acc', 0.0),
        filename='compare_final_acc_bar.png',
        title='Final Test Accuracy',
        ylabel='Accuracy',
    )
    return True


def save_strategy_summary_plot(summary_rows, output_path, title='Three Distillation Strategies'):
    if not plotting_available() or not summary_rows:
        return False

    import numpy as np

    labels = [row['display_label'] for row in summary_rows]
    best_means = [row['best_test_acc_mean'] for row in summary_rows]
    best_stds = [row['best_test_acc_std'] for row in summary_rows]
    final_means = [row['final_test_acc_mean'] for row in summary_rows]
    final_stds = [row['final_test_acc_std'] for row in summary_rows]

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#f7f3eb')
    ax.set_facecolor('#fffdf8')

    best_bars = ax.bar(
        x - width / 2,
        best_means,
        width,
        yerr=best_stds,
        color='#d98f3d',
        edgecolor='#8a5420',
        linewidth=1.0,
        capsize=6,
        label='Best Accuracy',
    )
    final_bars = ax.bar(
        x + width / 2,
        final_means,
        width,
        yerr=final_stds,
        color='#4d7ea8',
        edgecolor='#23405d',
        linewidth=1.0,
        capsize=6,
        label='Final Accuracy',
    )

    ax.set_title(title, fontsize=15, pad=16, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0.80, max(best_means) + 0.08)
    ax.grid(axis='y', alpha=0.22, linestyle='--')
    ax.legend(frameon=False, loc='upper left')

    _annotate_bars(ax, best_bars)
    _annotate_bars(ax, final_bars)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return True


def save_ablation_summary_plot(summary_rows, output_path, title='Ablation Summary'):
    if not plotting_available() or not summary_rows:
        return False

    import numpy as np

    labels = [row['label'] for row in summary_rows]
    final_means = [row['final_test_acc_mean'] for row in summary_rows]
    final_stds = [row['final_test_acc_std'] for row in summary_rows]
    best_means = [row['best_test_acc_mean'] for row in summary_rows]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#f7f3eb')
    ax.set_facecolor('#fffdf8')
    bars = ax.bar(
        x,
        final_means,
        yerr=final_stds,
        color=['#6c8ead', '#d98f3d', '#b56576', '#7aa974', '#8d6cab', '#c76d3a'],
        edgecolor='#3d3d3d',
        linewidth=0.9,
        capsize=6,
    )
    ax.plot(x, best_means, color='#2f4858', marker='o', linewidth=2.0, label='Best Accuracy Mean')

    ax.set_title(title, fontsize=15, pad=14, weight='bold')
    ax.set_ylabel('Final Accuracy', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha='right')
    ax.grid(axis='y', alpha=0.22, linestyle='--')
    ax.legend(frameon=False, loc='upper left')

    for bar, value in zip(bars, final_means):
        ax.annotate(
            '{:.2f}%'.format(value * 100.0),
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=9,
            color='#2f2f2f',
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return True


def _save_curve(x_values, y_values, output_path, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_values, y_values, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_detailed_curve(x_values, y_values, output_path, title, ylabel, line_color, fill_color):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#f7f3eb')
    ax.set_facecolor('#fffdf8')
    ax.plot(
        x_values,
        y_values,
        linewidth=2.4,
        marker='o',
        markersize=6,
        color=line_color,
        markerfacecolor='#fffaf0',
        markeredgewidth=1.2,
    )
    ax.fill_between(x_values, y_values, color=fill_color, alpha=0.65)
    ax.set_title(title, fontsize=15, weight='bold', pad=14)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.24, linestyle='--')
    ax.set_xticks(x_values)

    for x_value, y_value in zip(x_values, y_values):
        display_text = '{:.2f}%'.format(y_value * 100.0) if ylabel.lower() == 'accuracy' else '{:.4f}'.format(y_value)
        ax.annotate(
            display_text,
            xy=(x_value, y_value),
            xytext=(0, 8),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=9,
            color='#2f2f2f',
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _save_multi_curve(experiments, output_dir, metric_key, filename, title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    for experiment in experiments:
        metric_values = experiment['metrics'][metric_key]
        ax.plot(range(len(metric_values)), metric_values, linewidth=2, label=experiment['label'])
    ax.set_title(title)
    ax.set_xlabel('Round')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close(fig)


def _save_summary_bar(experiments, output_dir, metric_getter, filename, title, ylabel):
    labels = [experiment['label'] for experiment in experiments]
    values = [metric_getter(experiment) for experiment in experiments]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close(fig)


def _annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            '{:.2f}%'.format(height * 100.0),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=10,
            color='#2f2f2f',
        )
