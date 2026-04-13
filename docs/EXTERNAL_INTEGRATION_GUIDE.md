# 外部项目接入指南

这份文档面向“已有联邦学习项目”的接入，不要求对方使用本仓库的训练骨架。

当前项目统一采用：

- **文件夹级插件接入**

不再提供单文件插件交付路线。

## 1. 推荐复制的文件夹

如果你要把插件复制到其他项目，优先复制：

- `src/plugins/`

其中核心运行所需主要是：

- `src/plugins/base.py`
- `src/plugins/__init__.py`
- `src/plugins/fedfed_plugin.py`
- `src/plugins/feature_split.py`

这里的 `feature_split.py` 当前承担的是低维投影功能，历史文件名保留仅为兼容仓库结构。

## 2. 外部项目只需要改 3 个接入点

### 接入点 1：模型 forward

你的模型需要支持返回中间特征。

最小要求：

```python
pred, h = model(X, return_feature=True)
```

如果当前模型不支持，就增加一个可选参数：

```python
def forward(self, x, return_feature=False):
    ...
    if return_feature:
        return logits, h
    return logits
```

### 接入点 2：客户端本地训练循环

每轮训练开始前：

```python
plugin.on_round_start(current_lr, server_payload)
```

每个 batch：

```python
pred, loss = plugin.train_batch(X, y)
```

一轮训练结束后：

```python
aux = plugin.build_upload_payload()
```

如果不用插件，仍然走原有训练逻辑。

### 接入点 3：服务端聚合循环

在服务器侧，除了原有的 FedAvg 参数聚合之外，再补两步：

下发前：

```python
server_payload = server_plugin.build_broadcast_payload()
```

聚合后：

```python
server_plugin.aggregate_client_payloads(local_updates)
```

这里的 `local_updates` 里需要保留：

```python
{
    "weights": ...,
    "num_samples": ...,
    "aux": ...
}
```

## 3. 最小客户端伪代码

```python
from src.plugins.fedfed_plugin import FedFedClientPlugin

plugin = FedFedClientPlugin(options, model, gpu)

for round_i in range(num_rounds):
    plugin.on_round_start(current_lr, server_payload)
    for X, y in dataloader:
        pred, loss = plugin.train_batch(X, y)

    local_update = {
        "weights": copy.deepcopy(model.state_dict()),
        "num_samples": len(local_dataset),
        "aux": plugin.build_upload_payload(),
    }
```

## 4. 最小服务端伪代码

```python
from src.plugins.fedfed_plugin import FedFedServerPlugin

server_plugin = FedFedServerPlugin(options, gpu)

for round_i in range(num_rounds):
    payload = server_plugin.build_broadcast_payload()
    local_updates = run_clients(payload)

    global_weights = fedavg(local_updates)
    server_plugin.aggregate_client_payloads(local_updates)
```

## 5. 当前文件夹级插件的前提条件

外部项目至少要满足：

- 使用 PyTorch
- 模型可返回中间特征
- 本地训练循环可插入一个插件调用点
- 服务端聚合循环可插入一个辅助聚合点

## 6. 推荐的外部接入顺序

1. 先让模型支持 `return_feature=True`
2. 再接客户端插件
3. 最后接服务端插件

不要一上来就同时改所有地方，否则很难排查问题。

## 7. 当前仓库里的推荐插件实现

当前统一推荐使用：

- `fedfed_prototype`

它是当前项目的标准实现，也是文件夹级接入模式下的唯一主实现。
