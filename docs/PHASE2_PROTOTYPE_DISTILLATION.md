# 第二阶段改动说明：从单均值蒸馏升级为类原型蒸馏

本阶段目标是让 FedFed 风格插件更贴近真实分类任务接入场景。

## 1. 为什么从 `z_s mean` 改成 prototype

第一阶段上传的是一个客户端级别的单均值向量：

- 优点是简单、通信量小
- 缺点是把所有类别压成一个目标，信息过粗

对于真实的小型分类任务，更合理的做法是上传类原型：

- 每个类别一个低维 prototype
- 服务端按类别聚合
- 客户端按类别蒸馏

这样更符合真实部署中的判别结构，也更容易作为插件接到现有联邦学习项目里。

## 2. 本阶段完成的改动

### 2.1 客户端从“上传单均值”改为“上传类原型”

文件：
- `src/fed_client/client.py`

主要改动：

- `global_sensitive_feature` 改为 `global_prototypes`
- 新增 `set_global_prototypes(...)`
- 保留 `set_global_sensitive_feature(...)` 作为兼容别名
- 本地训练时按 batch 内类别统计 `z_s`
- 每轮训练结束后按类别生成本地 prototype
- 对每个 prototype 单独进行裁剪和加噪
- 上传格式从：
  - `aux["sensitive_feature"] = tensor`
  变为：
  - `aux["prototypes"] = {class_id: {"prototype": tensor, "count": int}}`

### 2.2 蒸馏损失从“对齐全局单向量”改为“对齐全局类原型”

文件：
- `src/fed_client/client.py`

新增：
- `_compute_prototype_distill_loss(...)`

行为：
- 遍历当前 batch 中出现的类别
- 如果该类别在服务端下发的全局 prototype bank 中存在
- 就计算本地该类 prototype 与全局该类 prototype 的 MSE
- 最后对 batch 中所有可对齐类别取平均

缺类处理：
- 如果客户端当前 batch 没有某类，跳过
- 如果服务端当前还没有某类全局原型，跳过

### 2.3 服务端从“聚合单均值”改为“聚合全局 prototype bank”

文件：
- `src/fed_server/fedbase.py`

主要改动：

- `global_sensitive_feature` 改为 `global_prototypes`
- 本地训练前下发 `client.set_global_prototypes(self.global_prototypes)`
- 聚合逻辑从 `_aggregate_aux_sensitive_feature(...)`
  改为 `_aggregate_aux_prototypes(...)`
- 服务端按类别、按样本数加权平均 prototype
- 聚合结果存成：
  - `self.global_prototypes = {class_id: prototype_tensor}`

## 3. 当前原型蒸馏流程

### 客户端侧

1. 前向得到中间特征 `h`
2. 通过 `FeatureSplitModule` 得到 `z_s`
3. 按 batch 内类别统计局部 prototype
4. 使用全局 prototype bank 计算蒸馏损失
5. 本轮结束后生成本地类原型并上传

### 服务端侧

1. 接收各客户端上传的类原型
2. 按类别、按样本数做加权聚合
3. 形成全局 prototype bank
4. 下一轮下发给客户端

## 4. 本阶段改动文件

- `src/fed_client/client.py`
- `src/fed_server/fedbase.py`
- `docs/PHASE2_PROTOTYPE_DISTILLATION.md`

## 5. 当前仍然保留的现实约束

本阶段仍然是“真实接入导向的模拟验证”，不是完整工程部署：

- 仍然是单机模拟 server/client
- 仍然没有网络通信层
- 仍然没有 secure aggregation
- 仍然没有多 prototype / 聚类中心

但相对第一阶段，当前设计已经更贴近真实联邦分类场景。

## 6. 下一阶段建议

后续更适合继续做这两件事：

1. 把 prototype 逻辑从 `client.py` / `fedbase.py` 中继续抽离成插件接口
2. 视目标任务是否存在类内多模态，再升级为“每类多个簇中心”
