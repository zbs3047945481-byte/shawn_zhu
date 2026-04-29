from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import torch
import copy
from src.plugins import build_client_plugin
from src.utils.tools import get_runtime_device
criterion = F.cross_entropy


class BaseClient():
    def __init__(self, options, id, local_dataset, model, optimizer, ):
        self.options = options
        self.id = id
        self.local_dataset = local_dataset
        self.model = model
        self.device = get_runtime_device(options)
        self.storage_device = torch.device('cpu')
        self.gpu = self.device.type != 'cpu'
        self.optimizer = optimizer

        self.model.to(self.storage_device)
        self.plugin = build_client_plugin(options, self.model, self.storage_device)
        self.plugin_payload = None

    def set_plugin_payload(self, payload):
        self.plugin_payload = payload

    def set_global_sensitive_feature(self, global_sensitive_feature):
        """Backward-compatible alias for legacy server hook names."""
        self.set_plugin_payload(global_sensitive_feature)

    def set_learning_rate(self, learning_rate):
        for group in self.optimizer.param_groups:
            group['lr'] = learning_rate

    def get_model_parameters(self):
        state_dict = self.model.state_dict()
        return state_dict

    def get_model_parameters_cpu(self):
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def set_model_parameters(self, model_parameters_dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_parameters_dict[key].detach().to(value.device)
        self.model.load_state_dict(state_dict)

    def local_train(self, ):
        begin_time = time.time()
        self._move_to_training_device()
        try:
            local_model_paras, return_dict, aux = self.local_update(self.local_dataset, self.options, )
        finally:
            self._move_to_storage_device()
        end_time = time.time()
        stats = {'id': self.id, "time": round(end_time - begin_time, 2)}
        stats.update(return_dict)
        # Update structure: weights (FedAvg) + num_samples + optional aux (FedFed)
        update = {"weights": local_model_paras, "num_samples": len(self.local_dataset), "aux": aux}
        return update, stats

    def local_update(self, local_dataset, options, ):
        use_plugin = self.plugin is not None
        pin_memory = self.gpu and options.get('dataloader_pin_memory', True)
        localTrainDataLoader = DataLoader(
            local_dataset,
            batch_size=options['batch_size'],
            shuffle=True,
            num_workers=max(int(options.get('dataloader_num_workers', 0)), 0),
            pin_memory=pin_memory,
        )
        self.model.train() #把模型设置为训练模式
        if use_plugin:
            self.plugin.on_round_start(self.optimizer.param_groups[0]['lr'], self.plugin_payload)#正式训练前，先把插件状态准备好。
        train_loss = train_acc = train_total = 0
        for epoch in range(options['local_epoch']):  #表示这个客户端会把自己的本地数据完整训练 local_epoch 遍。
            for X, y in localTrainDataLoader:
                if self.gpu:
                    X = X.to(self.device, non_blocking=pin_memory)
                    y = y.to(self.device, non_blocking=pin_memory)
                if use_plugin:
                    pred, loss = self.plugin.train_batch(X, y)
                else:
                    self.optimizer.zero_grad()
                    pred = self.model(X)
                    loss = criterion(pred, y)
                    loss.backward()
                    self.optimizer.step()
                _, predicted = torch.max(pred, 1) #从预测 logits 中取每个样本得分最高的类别，作为预测标签。
                correct = predicted.eq(y).sum().item() #统计当前 batch 中预测正确的样本数。
                target_size = y.size(0) #得到当前 batch 的样本数量。
                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size
        local_model_paras = self.get_model_parameters_cpu() #训练结束后，把上传权重固定到CPU，避免客户端模型常驻GPU。
        return_dict = {"id": self.id,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        if use_plugin and hasattr(self.plugin, 'collect_reference_prototypes'):
            self.plugin.collect_reference_prototypes(localTrainDataLoader)
        aux = self.plugin.build_upload_payload() if use_plugin else None
        return local_model_paras, return_dict, aux

    def _move_to_training_device(self):
        self.model.to(self.device)
        self._move_optimizer_state(self.optimizer, self.device)
        if self.plugin is not None and hasattr(self.plugin, 'to_device'):
            self.plugin.to_device(self.device)

    def _move_to_storage_device(self):
        self.model.to(self.storage_device)
        self._move_optimizer_state(self.optimizer, self.storage_device)
        if self.plugin is not None and hasattr(self.plugin, 'to_device'):
            self.plugin.to_device(self.storage_device)
        if self.gpu and self.device.type == 'cuda':
            torch.cuda.empty_cache()

    @staticmethod
    def _move_optimizer_state(optimizer, device):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
