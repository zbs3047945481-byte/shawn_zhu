import torch.nn as nn
import torch
from torchvision.models import resnet18


class CifarResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = resnet18(weights=None, num_classes=num_classes)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

    def forward(self, inputs, return_feature=False):
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        feature = torch.flatten(x, 1)
        logits = self.backbone.fc(feature)
        if return_feature:
            return logits, feature
        return logits

    def classify_feature(self, feature):
        return self.backbone.fc(feature)
