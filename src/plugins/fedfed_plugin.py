import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.plugins.feature_split import FeatureSplitModule
from src.plugins.base import BaseClientPlugin, BaseServerPlugin


mse_loss = nn.MSELoss()


class FedFedClientPlugin(BaseClientPlugin):
    def __init__(self, options, model, device):
        self.options = options
        self.model = model
        self.device = device
        self.gpu = device.type != 'cpu'
        self.global_prototypes = None
        self.global_prototype_counts = {}
        self.global_projection_state = None
        self.prototype_sums = {}
        self.prototype_counts = {}
        self.current_round = 0

        feature_dim = options.get('fedfed_feature_dim', 512)
        sensitive_dim = options.get('fedfed_sensitive_dim', 64)
        self.enable_projection = options.get('fedfed_enable_projection', True)
        self.enable_prototype_sharing = options.get('fedfed_enable_prototype_sharing', True)
        self.enable_distill = options.get('fedfed_enable_distill', True)
        self.enable_anchor = options.get('fedfed_enable_anchor', True)
        self.enable_clip = options.get('fedfed_enable_clip', True)
        self.enable_noise = options.get('fedfed_enable_noise', True)
        self.use_cosine_distill = options.get('fedfed_use_cosine_distill', True)
        self.normalize_prototypes = options.get('fedfed_normalize_prototypes', True)
        self.projection_module = None
        self.reference_model = None
        if self.enable_projection:
            self.projection_module = FeatureSplitModule(feature_dim, sensitive_dim)
            self.projection_module.to(self.device)

        trainable_params = list(self.model.parameters())
        if self.projection_module is not None:
            trainable_params += list(self.projection_module.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=options.get('lr', 0.001))

    def on_round_start(self, learning_rate, server_payload):
        for group in self.optimizer.param_groups:
            group['lr'] = learning_rate
        self.global_projection_state = None
        self.global_prototype_counts = {}
        if server_payload is not None:
            self.global_projection_state = server_payload.get('projection_state')
            self.current_round = int(server_payload.get('round_index', self.current_round))
        if self.enable_prototype_sharing:
            self.global_prototypes = None if server_payload is None else server_payload.get('global_prototypes')
            self.global_prototype_counts = {} if server_payload is None else server_payload.get('global_prototype_counts', {})
        else:
            self.global_prototypes = None
        if self.projection_module is not None and self.global_projection_state is not None:
            projection_state = {
                key: value.to(self.device)
                for key, value in self.global_projection_state.items()
            }
            self.projection_module.load_state_dict(projection_state, strict=True)
        if self.projection_module is not None:
            self.projection_module.train()
        if self.enable_anchor:
            self.reference_model = copy.deepcopy(self.model).to(self.device)
            self.reference_model.eval()
            for parameter in self.reference_model.parameters():
                parameter.requires_grad_(False)
        else:
            self.reference_model = None
        self.prototype_sums = {}
        self.prototype_counts = {}

    def _clip_and_noise(self, feature):
        clip_norm = self.options.get('fedfed_clip_norm', 1.0)
        noise_sigma = self.options.get('fedfed_noise_sigma', 0.1)

        if self.enable_clip and clip_norm is not None and clip_norm > 0:
            norm = feature.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.clamp(clip_norm / norm, max=1.0)
            feature = feature * scale
        if self.enable_noise and noise_sigma is not None and noise_sigma > 0:
            feature = feature + noise_sigma * torch.randn_like(feature)
        return feature

    def _normalize_prototype(self, prototype):
        if not self.normalize_prototypes:
            return prototype
        return F.normalize(prototype, dim=-1)

    def _count_reliability(self, count):
        tau = float(self.options.get('fedfed_distill_count_tau', 8.0))
        if tau <= 0:
            return 1.0
        return float(count) / (float(count) + tau)

    def _distill_strength(self):
        warmup_rounds = int(self.options.get('fedfed_distill_warmup_rounds', 0))
        if warmup_rounds <= 0:
            return float(self.options.get('fedfed_lambda_distill', 1.0))
        progress = min(max(self.current_round, 0) / float(warmup_rounds), 1.0)
        return float(self.options.get('fedfed_lambda_distill', 1.0)) * progress

    def _compute_anchor_loss(self, h, X):
        if not self.enable_anchor or self.reference_model is None:
            return None
        with torch.no_grad():
            _, reference_h = self.reference_model(X, return_feature=True)
        normalized_h = F.normalize(h, dim=-1)
        normalized_reference_h = F.normalize(reference_h, dim=-1)
        return 1.0 - F.cosine_similarity(normalized_h, normalized_reference_h, dim=-1).mean()

    def _compute_prototype_distill_loss(self, z_s, y):
        if not self.enable_distill or not self.global_prototypes:
            return None
        prototype_losses = []
        prototype_weights = []
        for label in y.unique():
            class_id = int(label.item())
            if class_id not in self.global_prototypes:
                continue
            class_mask = (y == label)
            class_count = int(class_mask.sum().item())
            local_proto = z_s[class_mask].mean(dim=0, keepdim=True)
            target_proto = self.global_prototypes[class_id].to(z_s.device).unsqueeze(0)
            local_proto = self._normalize_prototype(local_proto)
            target_proto = self._normalize_prototype(target_proto)
            if self.use_cosine_distill:
                loss_value = 1.0 - F.cosine_similarity(local_proto, target_proto, dim=-1).mean()
            else:
                loss_value = mse_loss(local_proto, target_proto)
            global_count = self.global_prototype_counts.get(class_id, class_count)
            reliability = self._count_reliability(class_count) * self._count_reliability(global_count)
            prototype_losses.append(loss_value)
            prototype_weights.append(loss_value.new_tensor(reliability))

        if not prototype_losses:
            return None
        loss_tensor = torch.stack(prototype_losses)
        weight_tensor = torch.stack(prototype_weights).clamp(min=1e-6)
        return (loss_tensor * weight_tensor).sum() / weight_tensor.sum()

    def train_batch(self, X, y):
        self.optimizer.zero_grad()
        pred, h = self.model(X, return_feature=True)
        z_s = self.projection_module(h) if self.projection_module is not None else h
        loss_cls = F.cross_entropy(pred, y)
        loss = loss_cls

        loss_anchor = self._compute_anchor_loss(h, X)
        if loss_anchor is not None:
            lambda_anchor = float(self.options.get('fedfed_lambda_anchor', 0.1))
            loss = loss + lambda_anchor * loss_anchor

        loss_distill = self._compute_prototype_distill_loss(z_s, y)
        if loss_distill is not None:
            lambda_distill = self._distill_strength()
            loss = loss + lambda_distill * loss_distill

        self._accumulate_batch_prototypes(self._normalize_prototype(z_s.detach()), y)
        loss.backward()
        self.optimizer.step()
        return pred, loss.detach()

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

    def build_upload_payload(self):
        payload = {}
        if self.projection_module is not None:
            payload['projection_state'] = {
                key: value.detach().cpu().clone()
                for key, value in self.projection_module.state_dict().items()
            }

        if self.enable_prototype_sharing and self.prototype_sums:
            local_prototypes = {}
            for class_id, feature_sum in self.prototype_sums.items():
                prototype = (feature_sum / self.prototype_counts[class_id]).unsqueeze(0)
                prototype = self._normalize_prototype(prototype)
                prototype = self._clip_and_noise(prototype).squeeze(0)
                local_prototypes[class_id] = {
                    'prototype': prototype.cpu(),
                    'count': self.prototype_counts[class_id],
                }
            payload['prototypes'] = local_prototypes

        return payload or None


class FedFedServerPlugin(BaseServerPlugin):
    def __init__(self, options, device):
        self.options = options
        self.device = device
        self.gpu = device.type != 'cpu'
        self.enable_prototype_sharing = options.get('fedfed_enable_prototype_sharing', True)
        self.normalize_prototypes = options.get('fedfed_normalize_prototypes', True)
        self.prototype_momentum = float(options.get('fedfed_prototype_momentum', 0.8))
        self.global_prototypes = None
        self.global_prototype_counts = {}
        self.global_projection_state = None
        self.current_round = 0

    def set_round_index(self, round_index):
        self.current_round = int(round_index)

    def _normalize_prototype(self, prototype):
        if not self.normalize_prototypes:
            return prototype
        return F.normalize(prototype.unsqueeze(0), dim=-1).squeeze(0)

    def build_broadcast_payload(self):
        payload = {}
        payload['round_index'] = self.current_round
        if self.global_projection_state is not None:
            payload['projection_state'] = self.global_projection_state
        if self.enable_prototype_sharing and self.global_prototypes is not None:
            payload['global_prototypes'] = self.global_prototypes
            payload['global_prototype_counts'] = self.global_prototype_counts
        return payload or None

    def aggregate_client_payloads(self, local_model_paras_set):
        projection_sums = {}
        projection_weight = 0
        prototype_sums = {}
        prototype_counts = {}
        for update in local_model_paras_set:
            aux = update.get('aux')
            if aux is None:
                continue
            num_sample = update['num_samples']
            if 'projection_state' in aux:
                if not projection_sums:
                    projection_sums = {
                        key: value.clone() * num_sample
                        for key, value in aux['projection_state'].items()
                    }
                else:
                    for key, value in aux['projection_state'].items():
                        projection_sums[key] += value * num_sample
                projection_weight += num_sample
            if not self.enable_prototype_sharing or 'prototypes' not in aux:
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

        if projection_sums:
            self.global_projection_state = {}
            for key, weighted_sum in projection_sums.items():
                averaged = weighted_sum / projection_weight
                self.global_projection_state[key] = averaged.to(self.device)
        else:
            self.global_projection_state = None

        if not self.enable_prototype_sharing:
            self.global_prototypes = None
            self.global_prototype_counts = {}
            return

        if not prototype_sums:
            return

        updated_prototypes = dict(self.global_prototypes or {})
        updated_counts = dict(self.global_prototype_counts or {})
        for class_id, weighted_sum in prototype_sums.items():
            prototype = weighted_sum / prototype_counts[class_id]
            prototype = self._normalize_prototype(prototype)
            if class_id in updated_prototypes and self.prototype_momentum > 0:
                prototype = self.prototype_momentum * updated_prototypes[class_id].to(prototype.device) + (
                    1.0 - self.prototype_momentum
                ) * prototype
                prototype = self._normalize_prototype(prototype)
            updated_prototypes[class_id] = prototype.to(self.device)
            updated_counts[class_id] = prototype_counts[class_id]
        self.global_prototypes = updated_prototypes
        self.global_prototype_counts = updated_counts
