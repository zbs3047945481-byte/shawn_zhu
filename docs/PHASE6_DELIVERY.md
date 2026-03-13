# 第六阶段改动说明：外部接入手册与单文件交付版本

本阶段目标是让当前插件更接近毕设最终交付物：

- 一份可以复制到其他联邦学习项目中的模块文件
- 一份面向外部项目的接入说明

## 1. 新增单文件版本

新增文件：

- `src/plugins/fedfed_single_file.py`

这个文件包含：

- `FeatureSplitModule`
- `FedFedSingleFileClientPlugin`
- `FedFedSingleFileServerPlugin`

设计目标：

- 尽量减少外部依赖
- 尽量适合复制到其他项目
- 更贴近“模块文件”式交付

## 2. 注册到当前插件系统

文件：

- `src/plugins/__init__.py`
- `src/options.py`

新增支持：

- `--plugin_name fedfed_single_file`

这样在当前仓库内也可以直接验证单文件版本，而不只是把它当静态交付物。

## 3. 新增外部接入手册

新增文件：

- `docs/EXTERNAL_INTEGRATION_GUIDE.md`

这份文档直接说明：

- 外部项目应该复制哪个文件
- 只需要改哪 3 个接入点
- 客户端和服务端最小伪代码长什么样

## 4. 当前仓库里的推荐用法

### 继续开发

优先用：

- `fedfed_prototype`

因为它结构更清晰，更适合继续迭代。

### 对外演示 / 交付

优先用：

- `fedfed_single_file`

因为它更符合“一个插件文件插进去就能用”的毕设表达方式。

## 5. 第六阶段的意义

到这一阶段，你已经不仅有：

- 可运行的插件协议
- 可运行的具体实现

还额外有了：

- 可复制的单文件版本
- 可执行的外部接入说明

这已经非常接近毕设最终展示形态。
