import torch
import torch.nn as nn


class FeatureSplitModule(nn.Module):
    def __init__(self, feature_dim, sensitive_dim):
        super(FeatureSplitModule, self).__init__()
        self.feature_dim = feature_dim
        self.sensitive_dim = sensitive_dim
        self.proj_s = nn.Linear(feature_dim, sensitive_dim)
        self.norm = nn.LayerNorm(sensitive_dim)

    def forward(self, h):
        z_s = self.proj_s(h)
        return self.norm(z_s)
