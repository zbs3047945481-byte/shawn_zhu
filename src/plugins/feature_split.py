import torch
import torch.nn as nn
#torch.nn 是 PyTorch 里专门用来构建神经网络的模块库。

class FeatureSplitModule(nn.Module):#nn.Module 是 PyTorch 里所有神经网络模块的基类。
    def __init__(self, feature_dim, sensitive_dim):
        super(FeatureSplitModule, self).__init__()
        self.feature_dim = feature_dim #原始特征维度
        self.sensitive_dim = sensitive_dim #敏感特征维度


        self.proj_s = nn.Linear(feature_dim, sensitive_dim) #原始特征到敏感特征的映射
#如果 h 是 [B, 512]，而 sensitive_dim=64，那输出就是 [B, 64]。
#weight：权重矩阵，决定输入的 512 个特征如何组合成输出的 64 个新特征
#bias：偏置向量，给每个输出维度额外加一个可学习的偏移量
#output = h @ weight.T + bias   矩阵运算

        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim, max(feature_dim // 4, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(feature_dim // 4, 1), 1),
        )#mlp：多层感知机（由若干个全连接层 Linear 加上非线性激活函数组成的小神经网络）


        self.proj_rev = nn.Linear(sensitive_dim, feature_dim)
#PyTorch 的 nn.Linear 默认作用在 tensor 的最后一维。
#也就是说它把最后一维当作特征维度

    def forward(self, h):
        z_s_lowdim = self.proj_s(h)  
        gate = torch.sigmoid(self.gate_mlp(h))  
        z_s_in_h_space = gate * self.proj_rev(z_s_lowdim)  
        z_r = h - z_s_in_h_space  

        return z_s_lowdim, z_r


#所以如果后面要把这个做成高质量论文，不能只说“分离了共享特征和鲁棒特征”，要补实验或约束来证明：
#目前对全局有影响的部分截断在z_s_lowdim = self.proj_s(h)  这行代码，下面的在自己过家家，没啥用。
