import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.plugins.base import BaseClientPlugin, BaseServerPlugin
except ImportError:
    class BaseClientPlugin:  # pragma: no cover
        pass

    class BaseServerPlugin:  # pragma: no cover
        pass


class FeatureSplitModule(nn.Module):
    """Self-contained feature split module for copy-paste delivery."""

    def __init__(self, feature_dim, sensitive_dim):
        super().__init__()
        self.proj_s = nn.Linear(feature_dim, sensitive_dim)
        hidden_dim = max(feature_dim // 4, 1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.proj_rev = nn.Linear(sensitive_dim, feature_dim)

    def forward(self, h):
        z_s = self.proj_s(h)
        gate = torch.sigmoid(self.gate_mlp(h))
        z_s_in_h = gate * self.proj_rev(z_s)
        z_r = h - z_s_in_h
        return z_s, z_r


class FedFedSingleFileClientPlugin(BaseClientPlugin):
    """Single-file delivery variant for external FL projects."""

    def __init__(self, options, model, gpu):
        self.options = options
        self.model = model
        self.gpu = gpu
        self.global_prototypes = None
        self.prototype_sums = {}
        self.prototype_counts = {}
        self.mse_loss = nn.MSELoss()

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
        self.feature_split_module.train()
        self.prototype_sums = {}
        self.prototype_counts = {}

    def train_batch(self, X, y):
        self.optimizer.zero_grad()
        pred, h = self.model(X, return_feature=True)
        z_s, _ = self.feature_split_module(h)
        loss_cls = F.cross_entropy(pred, y)
        loss = loss_cls

        loss_distill = self._compute_prototype_distill_loss(z_s, y)
        if loss_distill is not None:
            lambda_distill = self.options.get('fedfed_lambda_distill', 1.0)
            loss = loss_cls + lambda_distill * loss_distill

        self._accumulate_batch_prototypes(z_s.detach(), y)
        loss.backward()
        self.optimizer.step()
        return pred, loss.detach()

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

    def _compute_prototype_distill_loss(self, z_s, y):
        if not self.global_prototypes:
            return None

        losses = []
        for label in y.unique():
            class_id = int(label.item())
            if class_id not in self.global_prototypes:
                continue
            mask = (y == label)
            local_proto = z_s[mask].mean(dim=0, keepdim=True)
            target_proto = self.global_prototypes[class_id].to(z_s.device).unsqueeze(0)
            losses.append(self.mse_loss(local_proto, target_proto))

        if not losses:
            return None
        return torch.stack(losses).mean()

    def _accumulate_batch_prototypes(self, z_s, y):
        for label in y.unique():
            class_id = int(label.item())
            mask = (y == label)
            class_sum = z_s[mask].sum(dim=0)
            class_count = int(mask.sum().item())
            if class_id not in self.prototype_sums:
                self.prototype_sums[class_id] = class_sum
                self.prototype_counts[class_id] = class_count
            else:
                self.prototype_sums[class_id] += class_sum
                self.prototype_counts[class_id] += class_count

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


class FedFedSingleFileServerPlugin(BaseServerPlugin):
    """Single-file delivery variant for external FL projects."""

    def __init__(self, options, gpu):
        self.options = options
        self.gpu = gpu
        self.global_prototypes = None

    def build_broadcast_payload(self):
        if self.global_prototypes is None:
            return None
        return {'global_prototypes': self.global_prototypes}

    def aggregate_client_payloads(self, local_model_paras_set):
        prototype_sums = {}
        prototype_counts = {}
        for update in local_model_paras_set:
            aux = update.get('aux')
            if aux is None or 'prototypes' not in aux:
                continue
            for class_id, payload in aux['prototypes'].items():
                prototype = payload['prototype']
                count = payload['count']
                if class_id not in prototype_sums:
                    prototype_sums[class_id] = prototype * count
                    prototype_counts[class_id] = count
                else:
                    prototype_sums[class_id] += prototype * count
                    prototype_counts[class_id] += count

        if not prototype_sums:
            return

        self.global_prototypes = {}
        for class_id, weighted_sum in prototype_sums.items():
            prototype = weighted_sum / prototype_counts[class_id]
            self.global_prototypes[class_id] = prototype.cuda() if self.gpu else prototype
