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
        self.prototype_features = {}
        self.current_round = 0
        self.adaptive_control = {}

        feature_dim = options.get('fedfed_feature_dim', 512)
        sensitive_dim = options.get('fedfed_sensitive_dim', 64)
        self.enable_adaptive_control = options.get('fedfed_adaptive_control', False)
        self.enable_projection = options.get('fedfed_enable_projection', True)
        self.enable_prototype_sharing = options.get('fedfed_enable_prototype_sharing', True)
        self.prototype_source = str(options.get('fedfed_prototype_source', 'train')).lower()
        self.num_prototypes_per_class = max(int(options.get('fedfed_num_prototypes_per_class', 1)), 1)
        self.min_samples_per_prototype = max(int(options.get('fedfed_min_samples_per_prototype', 8)), 1)
        self.prototype_kmeans_iters = max(int(options.get('fedfed_prototype_kmeans_iters', 8)), 1)
        self.enable_distill = options.get('fedfed_enable_distill', True)
        self.enable_contrastive_distill = options.get('fedfed_enable_contrastive_distill', False)
        self.enable_anchor = options.get('fedfed_enable_anchor', True)
        self.enable_proto_cls = options.get('fedfed_enable_proto_cls', False)
        self.enable_clip = options.get('fedfed_enable_clip', True)
        self.enable_noise = options.get('fedfed_enable_noise', True)
        self.use_cosine_distill = options.get('fedfed_use_cosine_distill', True)
        self.normalize_prototypes = options.get('fedfed_normalize_prototypes', True)
        self.projection_module = None
        self.reference_projection_module = None
        self.reference_model = None
        if self.enable_projection:
            self.projection_module = FeatureSplitModule(feature_dim, sensitive_dim)
            self.projection_module.to(self.device)

        trainable_params = list(self.model.parameters())
        if self.projection_module is not None:
            trainable_params += list(self.projection_module.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=options.get('lr', 0.001))

    def to_device(self, device):
        self.device = device
        self.gpu = device.type != 'cpu'
        if self.projection_module is not None:
            self.projection_module.to(device)
        if self.reference_projection_module is not None:
            self.reference_projection_module.to(device)
        if self.reference_model is not None:
            self.reference_model.to(device)
        self._move_optimizer_state(device)
        self.prototype_sums = {
            class_id: feature_sum.to(device)
            for class_id, feature_sum in self.prototype_sums.items()
        }
        self.prototype_features = {
            class_id: feature_tensor.to(device)
            for class_id, feature_tensor in self.prototype_features.items()
        }
        if self.global_prototypes is not None:
            self.global_prototypes = {
                class_id: prototype.to(device)
                for class_id, prototype in self.global_prototypes.items()
            }
        if self.global_projection_state is not None:
            self.global_projection_state = {
                key: value.to(device)
                for key, value in self.global_projection_state.items()
            }

    def _move_optimizer_state(self, device):
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def on_round_start(self, learning_rate, server_payload):
        for group in self.optimizer.param_groups:
            group['lr'] = learning_rate
        self.global_projection_state = None
        self.global_prototype_counts = {}
        if server_payload is not None:
            self.global_projection_state = server_payload.get('projection_state')
            self.current_round = int(server_payload.get('round_index', self.current_round))
            self.adaptive_control = server_payload.get('adaptive_control', {}) or {}
        else:
            self.adaptive_control = {}
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
        needs_reference_model = self.enable_anchor or self.prototype_source == 'reference'
        if needs_reference_model:
            self.reference_model = copy.deepcopy(self.model).to(self.device)
            self.reference_model.eval()
            for parameter in self.reference_model.parameters():
                parameter.requires_grad_(False)
            if self.projection_module is not None:
                self.reference_projection_module = copy.deepcopy(self.projection_module).to(self.device)
                self.reference_projection_module.eval()
                for parameter in self.reference_projection_module.parameters():
                    parameter.requires_grad_(False)
            else:
                self.reference_projection_module = None
        else:
            self.reference_model = None
            self.reference_projection_module = None
        self.prototype_sums = {}
        self.prototype_counts = {}
        self.prototype_features = {}

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
        if self.enable_adaptive_control:
            max_lambda = float(self.options.get('fedfed_lambda_distill_max', self.options.get('fedfed_lambda_distill', 1.0)))
            scale = float(self.adaptive_control.get('distill_scale', 0.0))
            return max_lambda * min(max(scale, 0.0), 1.0)
        warmup_rounds = int(self.options.get('fedfed_distill_warmup_rounds', 0))
        if warmup_rounds <= 0:
            return float(self.options.get('fedfed_lambda_distill', 1.0))
        if str(self.options.get('fedfed_distill_warmup_mode', 'linear')).lower() == 'hard':
            if self.current_round < warmup_rounds:
                return 0.0
            return float(self.options.get('fedfed_lambda_distill', 1.0))
        progress = min(max(self.current_round, 0) / float(warmup_rounds), 1.0)
        return float(self.options.get('fedfed_lambda_distill', 1.0)) * progress

    def _anchor_strength(self, loss_anchor):
        if self.enable_adaptive_control:
            max_lambda = float(self.options.get('fedfed_lambda_anchor_max', self.options.get('fedfed_lambda_anchor', 0.1)))
            threshold = float(self.options.get('fedfed_anchor_drift_threshold', 0.08))
            slope = float(self.options.get('fedfed_anchor_drift_slope', 50.0))
            drift = float(loss_anchor.detach().item())
            gate = torch.sigmoid(loss_anchor.new_tensor(slope * (drift - threshold))).item()
            return max_lambda * gate
        if self.options.get('fedfed_anchor_epoch_scaling', False):
            max_lambda = float(self.options.get('fedfed_lambda_anchor_max', self.options.get('fedfed_lambda_anchor', 0.1)))
            ref_epoch = max(float(self.options.get('fedfed_anchor_ref_epoch', 5.0)), 1.0)
            local_epoch = max(float(self.options.get('local_epoch', 1)), 0.0)
            return max_lambda * min(local_epoch / ref_epoch, 1.0)
        return float(self.options.get('fedfed_lambda_anchor', 0.1))

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
            target_proto = self.global_prototypes[class_id].to(z_s.device)
            if target_proto.dim() == 1:
                target_proto = target_proto.unsqueeze(0)
            local_proto = self._normalize_prototype(local_proto)
            target_proto = self._normalize_prototype(target_proto)
            if self.use_cosine_distill:
                distances = 1.0 - F.cosine_similarity(local_proto, target_proto, dim=-1)
                loss_value = distances.min()
            else:
                distances = F.mse_loss(local_proto.expand_as(target_proto), target_proto, reduction='none').mean(dim=-1)
                loss_value = distances.min()
            global_count = self.global_prototype_counts.get(class_id, class_count)
            reliability = self._count_reliability(class_count) * self._count_reliability(global_count)
            prototype_losses.append(loss_value)
            prototype_weights.append(loss_value.new_tensor(reliability))

        if not prototype_losses:
            return None
        loss_tensor = torch.stack(prototype_losses)
        weight_tensor = torch.stack(prototype_weights).clamp(min=1e-6)
        return (loss_tensor * weight_tensor).sum() / weight_tensor.sum()

    def _compute_prototype_contrastive_loss(self, z_s, y):
        if not self.enable_contrastive_distill or not self.global_prototypes:
            return None

        labels = sorted(self.global_prototypes.keys())
        all_prototypes = []
        prototype_labels = []
        for class_id in labels:
            prototypes = self.global_prototypes[class_id].to(z_s.device)
            if prototypes.dim() == 1:
                prototypes = prototypes.unsqueeze(0)
            all_prototypes.append(prototypes)
            prototype_labels.extend([class_id] * prototypes.size(0))
        if not all_prototypes:
            return None

        prototype_matrix = self._normalize_prototype(torch.cat(all_prototypes, dim=0))
        prototype_label_tensor = torch.tensor(prototype_labels, device=z_s.device, dtype=torch.long)
        temperature = max(float(self.options.get('fedfed_contrastive_temperature', 0.2)), 1e-6)
        contrastive_losses = []
        contrastive_weights = []

        for label in y.unique():
            class_id = int(label.item())
            positive_mask = prototype_label_tensor == class_id
            negative_mask = ~positive_mask
            if not positive_mask.any() or not negative_mask.any():
                continue

            class_mask = y == label
            class_count = int(class_mask.sum().item())
            local_proto = self._normalize_prototype(z_s[class_mask].mean(dim=0, keepdim=True))
            logits = torch.mm(local_proto, prototype_matrix.t()).squeeze(0) / temperature
            positive_logit = logits[positive_mask].max()
            denominator = torch.logsumexp(logits, dim=0)
            loss_value = denominator - positive_logit

            global_count = self.global_prototype_counts.get(class_id, class_count)
            reliability = self._count_reliability(class_count) * self._count_reliability(global_count)
            contrastive_losses.append(loss_value)
            contrastive_weights.append(loss_value.new_tensor(reliability))

        if not contrastive_losses:
            return None
        loss_tensor = torch.stack(contrastive_losses)
        weight_tensor = torch.stack(contrastive_weights).clamp(min=1e-6)
        return (loss_tensor * weight_tensor).sum() / weight_tensor.sum()

    def _compute_proto_cls_loss(self, h, y):
        if not self.enable_proto_cls or not hasattr(self.model, 'classify_feature'):
            return None
        prototypes = []
        labels = []
        for label in y.unique():
            class_mask = (y == label)
            if not class_mask.any():
                continue
            prototypes.append(h[class_mask].mean(dim=0))
            labels.append(label)
        if not prototypes:
            return None
        prototype_tensor = torch.stack(prototypes, dim=0)
        label_tensor = torch.stack(labels).long().to(h.device)
        logits = self.model.classify_feature(prototype_tensor)
        return F.cross_entropy(logits, label_tensor)

    def train_batch(self, X, y):
        self.optimizer.zero_grad()
        pred, h = self.model(X, return_feature=True)
        z_s = self.projection_module(h) if self.projection_module is not None else h
        loss_cls = F.cross_entropy(pred, y)
        loss = loss_cls

        loss_anchor = self._compute_anchor_loss(h, X)
        if loss_anchor is not None:
            lambda_anchor = self._anchor_strength(loss_anchor)
            loss = loss + lambda_anchor * loss_anchor

        loss_distill = self._compute_prototype_distill_loss(z_s, y)
        if loss_distill is not None:
            lambda_distill = self._distill_strength()
            loss = loss + lambda_distill * loss_distill

        loss_contrastive = self._compute_prototype_contrastive_loss(z_s, y)
        if loss_contrastive is not None:
            lambda_contrastive = float(self.options.get('fedfed_lambda_contrastive', 0.05))
            loss = loss + lambda_contrastive * loss_contrastive

        loss_proto_cls = self._compute_proto_cls_loss(h, y)
        if loss_proto_cls is not None:
            lambda_proto_cls = float(self.options.get('fedfed_lambda_proto_cls', 0.1))
            loss = loss + lambda_proto_cls * loss_proto_cls

        if self.prototype_source == 'train':
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

    def _accumulate_batch_features(self, z_s, y):
        for label in y.unique():
            class_id = int(label.item())
            class_mask = (y == label)
            features = z_s[class_mask].detach()
            if class_id not in self.prototype_features:
                self.prototype_features[class_id] = features
                self.prototype_counts[class_id] = int(class_mask.sum().item())
            else:
                self.prototype_features[class_id] = torch.cat([self.prototype_features[class_id], features], dim=0)
                self.prototype_counts[class_id] += int(class_mask.sum().item())

    def _kmeans_prototypes(self, features, max_k):
        sample_count = int(features.size(0))
        if sample_count == 0:
            return None, None
        target_k = min(max_k, sample_count // self.min_samples_per_prototype)
        if target_k <= 1:
            return features.mean(dim=0, keepdim=True), features.new_tensor([sample_count], dtype=torch.float32)

        features = features.detach()
        init_indices = torch.linspace(0, sample_count - 1, steps=target_k, device=features.device).long()
        centers = features[init_indices].clone()
        assignments = None
        for _ in range(self.prototype_kmeans_iters):
            distances = torch.cdist(features, centers)
            assignments = distances.argmin(dim=1)
            new_centers = []
            for cluster_id in range(target_k):
                mask = assignments == cluster_id
                if mask.any():
                    new_centers.append(features[mask].mean(dim=0))
                else:
                    new_centers.append(centers[cluster_id])
            new_centers = torch.stack(new_centers, dim=0)
            if torch.allclose(new_centers, centers, rtol=1e-4, atol=1e-6):
                centers = new_centers
                break
            centers = new_centers
        if assignments is None:
            assignments = torch.cdist(features, centers).argmin(dim=1)
        counts = torch.stack([(assignments == cluster_id).sum() for cluster_id in range(target_k)]).to(
            device=features.device,
            dtype=torch.float32,
        )
        return centers, counts

    def collect_reference_prototypes(self, dataloader):
        if (
            not self.enable_prototype_sharing
            or self.prototype_source != 'reference'
            or self.reference_model is None
        ):
            return

        self.prototype_sums = {}
        self.prototype_counts = {}
        self.prototype_features = {}
        was_training = self.model.training
        self.reference_model.eval()
        if self.reference_projection_module is not None:
            self.reference_projection_module.eval()

        max_batches = int(self.options.get('fedfed_reference_proto_max_batches', 0))
        pin_memory = self.device.type != 'cpu' and self.options.get('dataloader_pin_memory', True)
        with torch.no_grad():
            for batch_index, (X, y) in enumerate(dataloader, start=1):
                if max_batches > 0 and batch_index > max_batches:
                    break
                if self.device.type != 'cpu':
                    X = X.to(self.device, non_blocking=pin_memory)
                    y = y.to(self.device, non_blocking=pin_memory)
                _, h = self.reference_model(X, return_feature=True)
                if self.reference_projection_module is not None:
                    z_s = self.reference_projection_module(h)
                else:
                    z_s = h
                z_s = self._normalize_prototype(z_s.detach())
                if self.num_prototypes_per_class > 1:
                    self._accumulate_batch_features(z_s, y)
                else:
                    self._accumulate_batch_prototypes(z_s, y)

        if was_training:
            self.model.train()

    def build_upload_payload(self):
        payload = {}
        if self.projection_module is not None:
            payload['projection_state'] = {
                key: value.detach().cpu().clone()
                for key, value in self.projection_module.state_dict().items()
            }

        if self.enable_prototype_sharing and self.prototype_features:
            local_prototypes = {}
            for class_id, features in self.prototype_features.items():
                prototypes, counts = self._kmeans_prototypes(features, self.num_prototypes_per_class)
                if prototypes is None:
                    continue
                prototypes = self._normalize_prototype(prototypes)
                prototypes = self._clip_and_noise(prototypes)
                local_prototypes[class_id] = {
                    'prototype': prototypes.cpu(),
                    'count': counts.cpu(),
                }
            if local_prototypes:
                payload['prototypes'] = local_prototypes
        elif self.enable_prototype_sharing and self.prototype_sums:
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
        self.num_prototypes_per_class = max(int(options.get('fedfed_num_prototypes_per_class', 1)), 1)
        self.prototype_kmeans_iters = max(int(options.get('fedfed_prototype_kmeans_iters', 8)), 1)
        self.global_prototypes = None
        self.global_prototype_counts = {}
        self.global_projection_state = None
        self.current_round = 0
        self.adaptive_control_state = {
            'distill_scale': 0.0,
            'prototype_stability': 0.0,
            'prototype_coverage': 0.0,
            'ready_rounds': 0,
        }

    def set_round_index(self, round_index):
        self.current_round = int(round_index)

    def _normalize_prototype(self, prototype):
        if not self.normalize_prototypes:
            return prototype
        return F.normalize(prototype.unsqueeze(0), dim=-1).squeeze(0)

    def _as_prototype_matrix(self, prototype):
        if prototype.dim() == 1:
            return prototype.unsqueeze(0)
        return prototype

    def _as_count_vector(self, count, prototype_count, device):
        if torch.is_tensor(count):
            count = count.to(device=device, dtype=torch.float32).view(-1)
        else:
            count = torch.tensor([float(count)], device=device)
        if count.numel() == prototype_count:
            return count
        if count.numel() == 1 and prototype_count > 1:
            return count.repeat(prototype_count) / float(prototype_count)
        return count[:prototype_count]

    def _weighted_kmeans(self, prototypes, counts, target_k):
        sample_count = int(prototypes.size(0))
        if sample_count == 0:
            return None, None
        target_k = min(max(target_k, 1), sample_count)
        if target_k == 1:
            total = counts.sum().clamp(min=1e-6)
            center = (prototypes * counts.unsqueeze(1)).sum(dim=0, keepdim=True) / total
            return center, counts.new_tensor([total])

        init_indices = torch.linspace(0, sample_count - 1, steps=target_k, device=prototypes.device).long()
        centers = prototypes[init_indices].clone()
        assignments = None
        for _ in range(self.prototype_kmeans_iters):
            distances = torch.cdist(prototypes, centers)
            assignments = distances.argmin(dim=1)
            new_centers = []
            for cluster_id in range(target_k):
                mask = assignments == cluster_id
                if mask.any():
                    cluster_counts = counts[mask]
                    total = cluster_counts.sum().clamp(min=1e-6)
                    new_centers.append((prototypes[mask] * cluster_counts.unsqueeze(1)).sum(dim=0) / total)
                else:
                    new_centers.append(centers[cluster_id])
            new_centers = torch.stack(new_centers, dim=0)
            if torch.allclose(new_centers, centers, rtol=1e-4, atol=1e-6):
                centers = new_centers
                break
            centers = new_centers
        if assignments is None:
            assignments = torch.cdist(prototypes, centers).argmin(dim=1)
        cluster_counts = torch.stack([
            counts[assignments == cluster_id].sum()
            for cluster_id in range(target_k)
        ])
        return centers, cluster_counts

    def _ema_match_prototypes(self, old_prototypes, new_prototypes):
        old_matrix = self._as_prototype_matrix(old_prototypes).to(new_prototypes.device)
        if old_matrix.size(0) == 1 and new_prototypes.size(0) == 1:
            return old_matrix
        distances = torch.cdist(new_prototypes, old_matrix)
        used_old = set()
        matched = []
        for new_index in range(new_prototypes.size(0)):
            order = torch.argsort(distances[new_index]).tolist()
            selected = None
            for old_index in order:
                if old_index not in used_old:
                    selected = old_index
                    break
            if selected is None:
                selected = order[0]
            used_old.add(selected)
            matched.append(old_matrix[selected])
        return torch.stack(matched, dim=0)

    def build_broadcast_payload(self):
        payload = {}
        payload['round_index'] = self.current_round
        if self.global_projection_state is not None:
            payload['projection_state'] = self.global_projection_state
        if self.enable_prototype_sharing and self.global_prototypes is not None:
            payload['global_prototypes'] = self.global_prototypes
            payload['global_prototype_counts'] = self.global_prototype_counts
        if self.options.get('fedfed_adaptive_control', False):
            payload['adaptive_control'] = dict(self.adaptive_control_state)
        return payload or None

    def _prototype_stability(self, old_prototypes, new_prototypes):
        if not old_prototypes or not new_prototypes:
            return 0.0
        similarities = []
        for class_id, prototype in new_prototypes.items():
            if class_id not in old_prototypes:
                continue
            old_proto = old_prototypes[class_id].to(prototype.device)
            new_proto = prototype
            if old_proto.dim() == 1:
                old_proto = old_proto.unsqueeze(0)
            if new_proto.dim() == 1:
                new_proto = new_proto.unsqueeze(0)
            if old_proto.size(0) != new_proto.size(0):
                min_count = min(old_proto.size(0), new_proto.size(0))
                old_proto = old_proto[:min_count]
                new_proto = new_proto[:min_count]
            old_proto = F.normalize(old_proto, dim=-1)
            new_proto = F.normalize(new_proto, dim=-1)
            similarities.append(F.cosine_similarity(old_proto, new_proto, dim=-1).mean().item())
        if not similarities:
            return 0.0
        return float(sum(similarities) / len(similarities))

    def _update_adaptive_control(self, old_prototypes, new_prototypes):
        if not self.options.get('fedfed_adaptive_control', False):
            return
        num_classes = max(int(self.options.get('fedfed_num_classes', 10)), 1)
        coverage = min(len(new_prototypes) / float(num_classes), 1.0) if new_prototypes else 0.0
        stability = self._prototype_stability(old_prototypes, new_prototypes)
        stability_threshold = float(self.options.get('fedfed_proto_stability_threshold', 0.90))
        coverage_threshold = float(self.options.get('fedfed_proto_coverage_threshold', 0.80))
        ramp_rounds = max(int(self.options.get('fedfed_adaptive_ramp_rounds', 3)), 1)
        is_ready = stability >= stability_threshold and coverage >= coverage_threshold
        ready_rounds = int(self.adaptive_control_state.get('ready_rounds', 0))
        ready_rounds = ready_rounds + 1 if is_ready else 0
        stability_gate = min(max((stability - stability_threshold) / max(1.0 - stability_threshold, 1e-6), 0.0), 1.0)
        coverage_gate = min(max((coverage - coverage_threshold) / max(1.0 - coverage_threshold, 1e-6), 0.0), 1.0)
        ramp_gate = min(ready_rounds / float(ramp_rounds), 1.0)
        self.adaptive_control_state = {
            'distill_scale': stability_gate * coverage_gate * ramp_gate,
            'prototype_stability': stability,
            'prototype_coverage': coverage,
            'ready_rounds': ready_rounds,
        }

    def aggregate_client_payloads(self, local_model_paras_set):
        projection_sums = {}
        projection_weight = 0
        prototype_sums = {}
        prototype_counts = {}
        prototype_candidates = {}
        prototype_candidate_counts = {}
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
                prototype = self._as_prototype_matrix(payload['prototype'].to(self.device))
                counts = self._as_count_vector(payload['count'], prototype.size(0), self.device)
                if prototype.size(0) > 1 or self.num_prototypes_per_class > 1:
                    if class_id not in prototype_candidates:
                        prototype_candidates[class_id] = [prototype]
                        prototype_candidate_counts[class_id] = [counts]
                    else:
                        prototype_candidates[class_id].append(prototype)
                        prototype_candidate_counts[class_id].append(counts)
                    continue
                count = counts.sum()
                if class_id not in prototype_sums:
                    prototype_sums[class_id] = prototype.squeeze(0) * count
                    prototype_counts[class_id] = count
                else:
                    prototype_sums[class_id] += prototype.squeeze(0) * count
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

        if not prototype_sums and not prototype_candidates:
            return

        old_prototypes = dict(self.global_prototypes or {})
        updated_prototypes = dict(self.global_prototypes or {})
        updated_counts = dict(self.global_prototype_counts or {})
        for class_id, weighted_sum in prototype_sums.items():
            prototype = weighted_sum / prototype_counts[class_id]
            prototype = self._normalize_prototype(prototype)
            if class_id in updated_prototypes and self.prototype_momentum > 0:
                old_proto = self._as_prototype_matrix(updated_prototypes[class_id].to(prototype.device)).squeeze(0)
                prototype = self.prototype_momentum * old_proto + (
                    1.0 - self.prototype_momentum
                ) * prototype
                prototype = self._normalize_prototype(prototype)
            updated_prototypes[class_id] = prototype.to(self.device)
            updated_counts[class_id] = float(prototype_counts[class_id].item())
        for class_id, candidates in prototype_candidates.items():
            prototypes = torch.cat(candidates, dim=0)
            counts = torch.cat(prototype_candidate_counts[class_id], dim=0)
            prototype, cluster_counts = self._weighted_kmeans(prototypes, counts, self.num_prototypes_per_class)
            if prototype is None:
                continue
            prototype = self._normalize_prototype(prototype)
            if class_id in updated_prototypes and self.prototype_momentum > 0:
                old_matched = self._ema_match_prototypes(updated_prototypes[class_id], prototype)
                prototype = self.prototype_momentum * old_matched + (1.0 - self.prototype_momentum) * prototype
                prototype = self._normalize_prototype(prototype)
            updated_prototypes[class_id] = prototype.to(self.device)
            updated_counts[class_id] = float(cluster_counts.sum().item())
        self.global_prototypes = updated_prototypes
        self.global_prototype_counts = updated_counts
        self._update_adaptive_control(old_prototypes, updated_prototypes)
