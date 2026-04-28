import json
import os

from tensorboardX import SummaryWriter

from src.plugins import resolve_plugin_name
from src.utils.plotting import save_single_run_plots


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


class Metrics(object):
    def __init__(self, options, clients, name=''):
        self.options = options

        num_rounds = options['round_num'] + 1
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}

        self.loss_on_g_test_data = [0] * num_rounds
        self.acc_on_g_test_data = [0] * num_rounds
        self.evaluated_rounds = []

        self.accumulation_delay = [0] * num_rounds
        self.accumulation_energy = [0] * num_rounds

        self.result_path = mkdir(os.path.join('./result', self.options['dataset_name']))
        suffix = '{}_sd{}_lr{}_ne{}_bs{}'.format(
            name,
            options['seed'],
            options['lr'],
            options['round_num'],
            options['batch_size'],
        )
        tag = options.get('experiment_tag', '').strip()
        self.exp_name = '{}_{}'.format(options['model_name'], suffix)
        if tag:
            self.exp_name = '{}_{}'.format(self.exp_name, tag)
        self.exp_dir = mkdir(os.path.join(self.result_path, self.exp_name))

        train_event_folder = mkdir(os.path.join(self.exp_dir, 'train.event'))
        test_event_folder = mkdir(os.path.join(self.exp_dir, 'eval.event'))
        self.train_writer = SummaryWriter(train_event_folder)
        self.eval_writer = SummaryWriter(test_event_folder)

    def update_test_stats(self, round_i, eval_stats):
        self.loss_on_g_test_data[round_i] = eval_stats['loss']
        self.acc_on_g_test_data[round_i] = eval_stats['acc']
        if round_i not in self.evaluated_rounds:
            self.evaluated_rounds.append(round_i)

        self.eval_writer.add_scalar('test_loss', eval_stats['loss'], round_i)
        self.eval_writer.add_scalar('test_acc', eval_stats['acc'], round_i)
        self._write_live()

    def _evaluated_series(self):
        rounds = sorted(self.evaluated_rounds)
        losses = [self.loss_on_g_test_data[round_i] for round_i in rounds]
        accs = [self.acc_on_g_test_data[round_i] for round_i in rounds]
        return rounds, losses, accs

    def _build_metrics(self):
        rounds, losses, accs = self._evaluated_series()
        return {
            'dataset': self.options['dataset_name'],
            'model_name': self.options['model_name'],
            'plugin_name': resolve_plugin_name(self.options) or 'none',
            'rounds': rounds,
            'loss_on_g_test_data': losses,
            'acc_on_g_test_data': accs,
            'best_test_acc': max(accs) if accs else None,
            'final_test_acc': accs[-1] if accs else None,
            'best_test_loss': min(losses) if losses else None,
            'final_test_loss': losses[-1] if losses else None,
            'final_round': rounds[-1] if rounds else None,
            'options': self.options,
        }

    def _write_table(self, table_dir):
        rounds, losses, accs = self._evaluated_series()
        with open(table_dir, 'w') as ouf:
            ouf.write('round,test_acc,test_loss\n')
            for round_i, acc_value, loss_value in zip(rounds, accs, losses):
                ouf.write('{},{:.10f},{:.10f}\n'.format(round_i, acc_value, loss_value))

    def _write_live(self):
        metrics_dir = os.path.join(self.exp_dir, 'live_metrics.json')
        with open(metrics_dir, 'w') as ouf:
            json.dump(self._build_metrics(), ouf, indent=2)
        self._write_table(os.path.join(self.exp_dir, 'live_metrics_table.csv'))

    def write(self):
        metrics = self._build_metrics()
        metrics_dir = os.path.join(self.exp_dir, 'metrics.json')

        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf, indent=2)

        self._write_table(os.path.join(self.exp_dir, 'metrics_table.csv'))

        save_single_run_plots(metrics, self.exp_dir)
