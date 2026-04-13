# 第三阶段改动说明：插件接口化重构

本阶段目标是让当前原型蒸馏能力从“嵌在联邦训练代码里的算法逻辑”变成“可以被联邦训练代码调用的插件模块”。

## 1. 重构目标

将以下逻辑从训练骨架中抽离：

- 客户端低维投影
- 客户端蒸馏损失计算
- 客户端类原型统计与上传载荷构造
- 服务端类原型聚合
- 服务端原型广播

重构后，`client.py` 和 `fedbase.py` 只保留训练与调度骨架。

## 2. 新增插件文件

新增文件：

- `src/plugins/fedfed_plugin.py`
- `src/plugins/__init__.py`

### `FedFedClientPlugin`

职责：

- 构建低维投影模块
- 维护插件专属优化器
- 接收服务端下发的 `global_prototypes`
- 计算 `L_cls + lambda * L_distill`
- 按类别统计本地 prototype
- 构造 `aux` 上传载荷

关键接口：

- `set_server_payload(payload)`
- `set_learning_rate(lr)`
- `train_mode()`
- `reset_round_state()`
- `zero_grad()`
- `compute_loss(X, y)`
- `step()`
- `build_upload_payload()`

### `FedFedServerPlugin`

职责：

- 维护服务端全局 prototype bank
- 从客户端更新中聚合类原型
- 构造下发给客户端的 payload

关键接口：

- `get_client_payload()`
- `aggregate_client_updates(local_model_paras_set)`

## 3. 训练骨架如何变化

### `src/fed_client/client.py`

现在的 `BaseClient` 不再直接知道：

- 低维投影模块的内部细节
- prototype distillation 的实现细节
- prototype 上传格式的内部构造

它现在只负责：

- 管理本地模型与基础优化器
- 在训练循环中判断插件是否存在
- 调用插件钩子完成前向、反向和上传

### `src/fed_server/fedbase.py`

现在的 `BaseFederated` 不再直接知道：

- 全局 prototype bank 怎么存
- prototype 怎么聚合
- prototype 广播格式是什么

它现在只负责：

- 维护 FedAvg 主流程
- 每轮训练前把服务端插件 payload 下发给客户端
- 每轮聚合后调用服务端插件处理 `aux`

## 4. 当前架构的意义

这一步的意义不是“彻底独立于任何 FL 项目”，而是先达到：

**模块级独立**

也就是：

- 训练框架保留
- 算法细节收敛到单独插件文件
- 后续可以继续向“文件夹级可插拔交付”收缩

这更符合毕设最终交付路径。

## 5. 当前代码中的插件边界

### 客户端骨架

- `src/fed_client/client.py`

### 服务端骨架

- `src/fed_server/fedbase.py`

### 插件核心

- `src/plugins/fedfed_plugin.py`

## 6. 下一步建议

如果继续沿毕设“即插即用模块”路线推进，下一步最合适的是：

1. 定义统一插件协议
   - 例如 `BaseClientPlugin` / `BaseServerPlugin`
2. 把 `FedFedClientPlugin` / `FedFedServerPlugin` 做成规范实现
3. 进一步减少 `client.py` / `fedbase.py` 对插件具体名称的感知
4. 最终收敛到一个可复制到其他 FL 项目中的模块文件
