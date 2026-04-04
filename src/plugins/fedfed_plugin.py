import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.feature_split import FeatureSplitModule
from src.plugins.base import BaseClientPlugin, BaseServerPlugin


mse_loss = nn.MSELoss()


class FedFedClientPlugin(BaseClientPlugin):
    def __init__(self, options, model, gpu):
        self.options = options
        self.model = model
        self.gpu = gpu
        self.global_prototypes = None
        self.prototype_reliability = None
        self.prototype_sums = {}
        self.prototype_counts = {}

        feature_dim = options.get('fedfed_feature_dim', 512)
        sensitive_dim = options.get('fedfed_sensitive_dim', 64)
        self.feature_split_module = FeatureSplitModule(feature_dim, sensitive_dim)
        if self.gpu:
            self.feature_split_module.cuda()

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.feature_split_module.parameters()),
            lr=options.get('lr', 0.001),
        )

    def on_round_start(self, learning_rate, server_payload):
        for group in self.optimizer.param_groups:
            group['lr'] = learning_rate
        self.global_prototypes = None if server_payload is None else server_payload.get('global_prototypes')
        self.prototype_reliability = None if server_payload is None else server_payload.get('prototype_reliability')
        self.feature_split_module.train()
        self.prototype_sums = {}
        self.prototype_counts = {}

    def _clip_and_noise(self, feature):
        clip_norm = self.options.get('fedfed_clip_norm', 1.0)
        noise_sigma = self.options.get('fedfed_noise_sigma', 0.1)

        if clip_norm is not None and clip_norm > 0:
            norm = feature.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.clamp(clip_norm / norm, max=1.0)
            feature = feature * scale
        if noise_sigma is not None and noise_sigma > 0:
            feature = feature + noise_sigma * torch.randn_like(feature, device=feature.device)
        return feature

    def _compute_prototype_distill_loss(self, z_s, y):
        if not self.global_prototypes: #如果服务器没有全局prototype，直接不蒸馏
            return None
        prototype_losses = [] #每个类别对应的 prototype loss
        prototype_weights = [] #每个类别对应的 reliability 权重
        for label in y.unique():
            class_id = int(label.item())
            if class_id not in self.global_prototypes: 
                continue  #如果服务器没有这个类别的全局prototype，跳过
            reliability = self._get_class_reliability(class_id)
            if reliability < self.options.get('fedfed_reliability_min', 0.05):
                continue  #如果这个类别的 reliability 权重低于阈值，跳过
            class_mask = (y == label) #根据标签，找到当前batch中属于这个类别的样本索引
            local_proto = z_s[class_mask].mean(dim=0, keepdim=True)   #计算这个类别在当前batch中的本地prototype
            target_proto = self.global_prototypes[class_id].to(z_s.device).unsqueeze(0)
    #取服务器下发的目标 prototype
            prototype_losses.append(mse_loss(local_proto, target_proto))
    #计算这个类别的 prototype loss
            prototype_weights.append(reliability)#把这个类别的 reliability 权重添加到列表中

        if not prototype_losses: #如果所有类别的 prototype loss 都为0，直接返回None
            return None
        loss_tensor = torch.stack(prototype_losses) #把所有类别的 prototype loss 堆叠成一个tensor
        weight_tensor = torch.tensor(prototype_weights, dtype=loss_tensor.dtype, device=loss_tensor.device) #把所有类别的 reliability 权重转换成一个tensor
        if weight_tensor.sum() <= 0: #如果所有类别的 reliability 权重之和为0，直接返回None
            return None
        return (loss_tensor * weight_tensor).sum() / weight_tensor.sum() #计算加权平均的 prototype loss

    def _get_class_reliability(self, class_id):
        if not self.options.get('fedfed_enable_reliability_gating', True):
            return 1.0
        if not self.prototype_reliability:
            return 0.0
        return float(self.prototype_reliability.get(class_id, 0.0))

    def train_batch(self, X, y):
        self.optimizer.zero_grad() #清空梯度
        pred, h = self.model(X, return_feature=True) #调用主模型 Mnist_CNN 的前向传播
        z_s, _ = self.feature_split_module(h) #调用特征分割模块，把特征分成敏感特征和非敏感特征
        loss_cls = F.cross_entropy(pred, y) #计算分类损失
        loss = loss_cls

        loss_distill = self._compute_prototype_distill_loss(z_s, y)
        if loss_distill is not None:
            lambda_distill = self.options.get('fedfed_lambda_distill', 1.0)
            loss = loss_cls + lambda_distill * loss_distill

        self._accumulate_batch_prototypes(z_s.detach(), y)
        loss.backward()
        self.optimizer.step()
        return pred, loss.detach()

#在客户端本地训练过程中，按类别累计当前轮看到的特征和样本数，为后面计算“每个类别的本地原型”做准备。
    def _accumulate_batch_prototypes(self, z_s, y):
        for label in y.unique():
            class_id = int(label.item())
            class_mask = (y == label)
            class_feature_sum = z_s[class_mask].sum(dim=0)
            class_count = int(class_mask.sum().item())
            if class_id not in self.prototype_sums:
                self.prototype_sums[class_id] = class_feature_sum
                self.prototype_counts[class_id] = class_count
            else:
                self.prototype_sums[class_id] += class_feature_sum
                self.prototype_counts[class_id] += class_count

#把客户端当前轮累计得到的各类别本地原型整理成可上传给服务器的 payload。
    def build_upload_payload(self):
        if not self.prototype_sums:
            return None
        local_prototypes = {}
        for class_id, feature_sum in self.prototype_sums.items():
            prototype = (feature_sum / self.prototype_counts[class_id]).unsqueeze(0)
            prototype = self._clip_and_noise(prototype).squeeze(0)
            local_prototypes[class_id] = {
                'prototype': prototype.cpu(),
                'count': self.prototype_counts[class_id],
            }
        return {'prototypes': local_prototypes}


class FedFedServerPlugin(BaseServerPlugin):
    def __init__(self, options, gpu):
        self.options = options
        self.gpu = gpu
        self.global_prototypes = None
        self.prototype_reliability = None

    def build_broadcast_payload(self):
        if self.global_prototypes is None:
            return None
        return {
            'global_prototypes': self.global_prototypes,
            'prototype_reliability': self.prototype_reliability,
        }

    def aggregate_client_payloads(self, local_model_paras_set):
        prototype_sums = {} #存某个类别的加权原型和
        prototype_counts = {} #存某个类别的样本数
        prototype_client_counts = {} #存某个类别的客户端样本数
        for update in local_model_paras_set:
            aux = update.get('aux')
            if aux is None or 'prototypes' not in aux: #如果客户端没有上传prototype，跳过
                continue
            for class_id, payload in aux['prototypes'].items():#遍历一个客户端上传的每个类别原型
                prototype = payload['prototype'] #取出这个类别的原型
                count = payload['count'] #取出这个类别的样本数
                if class_id not in prototype_sums:
                    prototype_sums[class_id] = prototype * count
                    prototype_counts[class_id] = count #存某个类别的样本数
                    prototype_client_counts[class_id] = 1 #存某个类别的客户端样本数
                else:
                    prototype_sums[class_id] += prototype * count #累加这个类别的加权原型和
                    prototype_counts[class_id] += count #累加这个类别的样本数
                    prototype_client_counts[class_id] += 1

        if not prototype_sums:
            return

        self.global_prototypes = {} #存所有类别的全局原型
        self.prototype_reliability = {} #存所有类别的reliability权重
        for class_id, weighted_sum in prototype_sums.items():
            prototype = weighted_sum / prototype_counts[class_id] #计算这个类别的全局原型
            self.global_prototypes[class_id] = prototype.cuda() if self.gpu else prototype #把全局原型存下来
            self.prototype_reliability[class_id] = self._compute_reliability(
                prototype_counts[class_id], #计算这个类别的样本数
                prototype_client_counts[class_id], #计算这个类别的客户端样本数
            ) #计算这个类别的reliability权重

    def _compute_reliability(self, sample_count, client_count): #计算这个类别的可靠性权重
        if not self.options.get('fedfed_enable_reliability_gating', True):
            return 1.0
        count_tau = max(float(self.options.get('fedfed_reliability_count_tau', 128.0)), 1.0) #计算这个类别的样本数阈值
        client_tau = max(float(self.options.get('fedfed_reliability_client_tau', 5.0)), 1.0) #计算这个类别的客户端样本数阈值
        count_reliability = min(float(sample_count) / count_tau, 1.0) #计算这个类别的样本数可靠性
        client_reliability = min(float(client_count) / client_tau, 1.0) #计算这个类别的客户端样本数可靠性
        return count_reliability * client_reliability #计算这个类别的可靠性权重
