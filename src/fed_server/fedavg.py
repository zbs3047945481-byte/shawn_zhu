#FedAvgTrainer=联邦服务器（Server）的控制器，负责：
#选客户端 → 下发模型 → 收回本地模型 → 聚合 → 测试 → 调学习率
#它不是具体训练模型的地方，而是“指挥谁训练、什么时候聚合”的地方


from src.fed_server.fedbase import BaseFederated
from torch import optim
from src.optimizers.adam import MyAdam
import numpy as np
from src.models.models import choose_model
from src.plugins import resolve_plugin_name

class FedAvgTrainer(BaseFederated):
    #options：实验配置（超参数）
    #dataset：全局数据对象（train/test）
    #clients_label：每个client的数据索引
    def __init__(self, options, dataset, clients_label):
        #根据 options['model_name'] 选择模型
        model = choose_model(options)
        model_builder = lambda: choose_model(options)
        self.move_model_to_gpu(model, options)

        #使用自定义Adam，本质等价于 torch.optim.Adam，这个优化器会被传给客户端用于本地训练
        optimizer_builder = lambda params: MyAdam(params, lr=options['lr'])
        self.optimizer = optimizer_builder(model.parameters())
        super(FedAvgTrainer, self).__init__(
            options,
            dataset,
            clients_label,#每个客户端对应的数据索引列表
            model,
            self.optimizer,
            model_builder=model_builder,
            optimizer_builder=optimizer_builder,
        )
        plugin_name = resolve_plugin_name(options)
        if plugin_name is not None:
            print('>>> Plugin ENABLED ({}, lambda_fd={}, lambda_shared={}, shared_buffer={})'.format(
                plugin_name,
                options.get('fedfed_lambda_fd', 1.0),
                options.get('fedfed_lambda_shared', 1.0),
                options.get('fedfed_shared_buffer_size', 800)))

    def train(self):
        print('>>> Select {} clients per round \n'.format(self._resolve_num_clients_per_round()))
        self._maybe_run_fedfed_two_stage()

        # self.latest_global_model = self.get_model_parameters()
        #self.num_round=通信轮数
        #每一轮=一次完整的：下发模型、本地训练、聚合
        early_stop = self._init_early_stop_state()
        stopped_early = False
        for round_i in range(self.num_round):
            #用当前全局模型、在测试集上评估、记录accuracy/loss
            eval_stats = self.test_latest_model_on_testdata(round_i)#全局模型评估
            if self._should_stop_early(round_i, eval_stats, early_stop):
                print('>>> Early stop at round {}: best_acc={:.3%}, current_acc={:.3%}'.format(
                    round_i, early_stop['best_acc'], eval_stats['acc']))
                stopped_early = True
                break
            #客户端抽样
            selected_clients = self.select_clients()

            #对每个client：1.下发全局模型参数；2.在client本地数据上训练local_epoch;3.收集训练后的模型参数
            #返回所有client的模型参数（通常是state_dict）和本地loss、样本数等统计信息
            local_model_paras_set, stats = self.local_train(round_i, selected_clients)

            #对所有客户端模型参数按样本数加权平均
            #得到新的全局模型参数
            self.latest_global_model = self.aggregate_parameters(local_model_paras_set)

            #根据轮数调整学习率；不是FedAvg必需步骤；但有助于收敛稳定
            self.optimizer.adjust_learning_rate(round_i)

        #最后一轮测试，把所有指标写入文件
        if not stopped_early:
            self.test_latest_model_on_testdata(self.num_round)
        self.metrics.write()

    #非常标准的FedAvg实现
    def select_clients(self):
        #计算本轮客户端数量，防止抽样数量大于总客户端数
        num_clients = self._resolve_num_clients_per_round()
        #随机选client索引，replace=False：不重复抽样，每轮client是随机的
        index = np.random.choice(len(self.clients), num_clients, replace=False,)
        #根据索引拿到client对象
        #self.clients 是在 BaseFederated 中初始化的
        #每个元素是一个 client 实例
        select_clients = []
        for i in index:
            select_clients.append(self.clients[i])
        #返回client列表
        return select_clients

    def _resolve_num_clients_per_round(self):
        if self.clients_num <= 0:
            raise ValueError('No clients are available for federation.')
        raw_num_clients = int(self.per_round_c_fraction * self.clients_num)
        return min(max(raw_num_clients, 1), self.clients_num)

    def _maybe_run_fedfed_two_stage(self):
        if self.server_plugin is None:
            return
        if self.options.get('plugin_name') != 'fedfed_image':
            return
        if not bool(self.options.get('fedfed_two_stage', True)):
            return
        if not hasattr(self.server_plugin, 'build_feature_distill_payload'):
            return

        distill_rounds = max(int(self.options.get('fedfed_distill_rounds', 15)), 0)
        print('>>> FedFed two-stage: feature distillation rounds = {}'.format(distill_rounds))
        for round_i in range(distill_rounds):
            selected_clients = self.select_clients()
            self.server_plugin.set_round_index(round_i)
            payload = self.server_plugin.build_feature_distill_payload()
            local_updates = []
            for i, client in enumerate(selected_clients, start=1):
                client.set_model_parameters(self.latest_global_model)
                client.set_learning_rate(self.optimizer.param_groups[0]['lr'])
                update, stat = client.plugin_feature_distill(payload)
                local_updates.append(update)
                print("Distill: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s ".format(
                       round_i, client.id, i, len(selected_clients),
                       stat['loss'], stat['acc'] * 100, stat['time'], ))
            self.latest_global_model = self._aggregate_weights_only(local_updates)
            self.server_plugin.aggregate_generator_states(local_updates)

        self._collect_fedfed_shared_dataset()

    def _collect_fedfed_shared_dataset(self):
        print('>>> FedFed two-stage: collecting shared performance-sensitive features')
        self.server_plugin.reset_shared_buffer()
        payload = self.server_plugin.build_feature_distill_payload()
        local_updates = []
        for client in self.clients:
            client.set_model_parameters(self.latest_global_model)
            client.set_learning_rate(self.optimizer.param_groups[0]['lr'])
            local_updates.append(client.plugin_collect_shared_features(payload))
        self.server_plugin.collect_shared_payloads(local_updates)

    def _aggregate_weights_only(self, local_model_paras_set):
        averaged_paras = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }
        train_data_num = 0
        for var in averaged_paras:
            if averaged_paras[var].is_floating_point():
                averaged_paras[var].zero_()
        for update in local_model_paras_set:
            num_sample = update["num_samples"]
            local_model_paras = update["weights"]
            for var in averaged_paras:
                if averaged_paras[var].is_floating_point():
                    averaged_paras[var] += num_sample * local_model_paras[var].to(averaged_paras[var].device)
            train_data_num += num_sample
        for var in averaged_paras:
            if averaged_paras[var].is_floating_point():
                averaged_paras[var] /= train_data_num
            else:
                largest_update = max(local_model_paras_set, key=lambda update: update["num_samples"])
                averaged_paras[var] = largest_update["weights"][var].clone().to(averaged_paras[var].device)
        return averaged_paras

    def _init_early_stop_state(self):
        return {
            'enabled': bool(self.options.get('early_stop_enable', False)),
            'min_rounds': max(int(self.options.get('early_stop_min_rounds', 0)), 0),
            'patience': max(int(self.options.get('early_stop_patience', 0)), 0),
            'min_delta': max(float(self.options.get('early_stop_min_delta', 0.0)), 0.0),
            'best_acc': -np.inf,
            'stale_rounds': 0,
        }

    def _should_stop_early(self, round_i, eval_stats, state):
        if not state['enabled'] or state['patience'] <= 0:
            return False
        current_acc = eval_stats['acc']
        if current_acc > state['best_acc'] + state['min_delta']:
            state['best_acc'] = current_acc
            state['stale_rounds'] = 0
            return False
        state['stale_rounds'] += 1
        return round_i >= state['min_rounds'] and state['stale_rounds'] >= state['patience']
