import torch
import torch.nn as nn
import torch.nn.functional as F

from src.plugins.feature_split import FeatureSplitModule
from src.plugins.base import BaseClientPlugin, BaseServerPlugin


mse_loss = nn.MSELoss()


class FedFedClientPlugin(BaseClientPlugin):
    def __init__(self, options, model, gpu):
        self.options = options
        self.model = model
        self.gpu = gpu
        self.global_prototypes = None
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
#在 PyTorch 里，一个优化器可以有多个参数组，不同参数组可以设置不同学习率、权重衰减等超参数。
            group['lr'] = learning_rate
#把每个参数组的学习率都更新成服务器当前下发的学习率。
#含义是：插件这部分参数更新节奏要和服务器主调度保持一致。
        self.global_prototypes = None if server_payload is None else server_payload.get('global_prototypes')
        self.feature_split_module.train()
        self.prototype_sums = {}
        self.prototype_counts = {}

    def _clip_and_noise(self, feature): #隐私保护
        clip_norm = self.options.get('fedfed_clip_norm', 1.0)
        noise_sigma = self.options.get('fedfed_noise_sigma', 0.1)

        if clip_norm is not None and clip_norm > 0:
            norm = feature.norm(dim=-1, keepdim=True).clamp(min=1e-8)
#在机器学习里，范数通常代表一个向量的“大小”“强度”“幅度”。
#对于特征向量来说，范数越大，通常说明这个特征整体激活越强。
#范数可以用来衡量一个特征是不是“过大”。
            scale = torch.clamp(clip_norm / norm, max=1.0)
#把 L2 范数限制在 1.0 以内，意思是：不允许上传的 prototype 特征向量太大。
#降低单个客户端影响力；提升训练稳定性；为加噪隐私保护做准备
#范数裁剪就是：如果一个向量太长，就把它按比例缩短；如果它本来不长，就保持不变。
#在 PyTorch 里，神经网络特征通常把“特征维度”放在最后一维。
            feature = feature * scale
        if noise_sigma is not None and noise_sigma > 0:
            feature = feature + noise_sigma * torch.randn_like(feature, device=feature.device)
#生成一个和 feature 形状一样的随机张量，每个元素来自标准正态分布：N(0, 1)，
        return feature
#Tensor 是数据结构，是 PyTorch 里的数据容器。


    def _compute_prototype_distill_loss(self, z_s, y): #计算蒸馏损失，本地原型对齐全局原型
        if not self.global_prototypes: #如果服务器没有全局prototype，直接不蒸馏
            return None
        prototype_losses = [] #保存每个类别的prototype对齐损失的列表
        for label in y.unique():
            class_id = int(label.item())
            if class_id not in self.global_prototypes: 
                continue  #如果服务器没有这个类别的全局prototype，跳过
            class_mask = (y == label) #根据标签，找到当前batch中属于这个类别的样本索引
            local_proto = z_s[class_mask].mean(dim=0, keepdim=True)   #计算这个类别在当前batch中的本地prototype
#keepdim=True 的作用是：求均值以后保留被压缩的维度。
            target_proto = self.global_prototypes[class_id].to(z_s.device).unsqueeze(0)
    #取服务器下发的目标 prototype
            prototype_losses.append(mse_loss(local_proto, target_proto))
    #计算这个类别的 prototype loss

        if not prototype_losses: #如果所有类别的 prototype loss 都为0，直接返回None
            return None
        loss_tensor = torch.stack(prototype_losses) #把所有类别的 prototype loss 堆叠成一个tensor
        return loss_tensor.mean() #计算所有可用类别 prototype loss 的平均值

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
            class_id = int(label.item())#把 PyTorch tensor 类型的类别标签转成 Python 整数。
            class_mask = (y == label)
            class_feature_sum = z_s[class_mask].sum(dim=0) #沿着第0维做操作
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
            prototype = (feature_sum / self.prototype_counts[class_id]).unsqueeze(0)#在最前面增加一个 batch 维度
            prototype = self._clip_and_noise(prototype).squeeze(0)#把刚才加上的 batch 维度去掉
            local_prototypes[class_id] = {
                'prototype': prototype.cpu(),#这一项保存当前类别的本地 prototype，.cpu() 的意思是把 tensor 移到 CPU 内存上。
                'count': self.prototype_counts[class_id],
            }#字典
        return {'prototypes': local_prototypes}


class FedFedServerPlugin(BaseServerPlugin):
    def __init__(self, options, gpu):
        self.options = options
        self.gpu = gpu
        self.global_prototypes = None

    def build_broadcast_payload(self):
        if self.global_prototypes is None:
            return None
        return {'global_prototypes': self.global_prototypes}

    def aggregate_client_payloads(self, local_model_paras_set):#本轮所有参与客户端上传的更新集合
        prototype_sums = {} #存某个类别的加权原型和
        prototype_counts = {} #存某个类别的样本数
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
                else:
                    prototype_sums[class_id] += prototype * count #累加这个类别的加权原型和
                    prototype_counts[class_id] += count #累加这个类别的样本数

        if not prototype_sums:
            return

        self.global_prototypes = {} #存所有类别的全局原型
        for class_id, weighted_sum in prototype_sums.items():
            prototype = weighted_sum / prototype_counts[class_id] #计算这个类别的全局原型
            self.global_prototypes[class_id] = prototype.cuda() if self.gpu else prototype #把全局原型存下来
