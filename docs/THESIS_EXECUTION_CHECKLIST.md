# 毕设实验执行清单

这份清单的目标不是解释实验原理，而是告诉你：

- 先跑什么
- 后跑什么
- 每组实验的用途
- 跑完后放进论文哪里

建议严格按顺序执行，避免一开始就把所有实验同时铺开。

## 1. 执行顺序总览

### 第一优先级

先跑：

```bash
python run_experiment_suite.py --suite thesis_main
```

用途：

- 这是整篇论文最核心的一组实验
- 用来证明你的方法在不同 `Non-IID` 强度下优于 `FedAvg`

论文位置：

- 主实验结果章节

输出重点：

- 对比曲线图
- 最终精度柱状图
- `suite_summary.csv`

### 第二优先级

再跑：

```bash
python run_experiment_suite.py --suite thesis_ablation
```

用途：

- 证明你的设计不是“随便加个模块就有效”
- 重点看 prototype 设计、噪声项、蒸馏权重、不同敏感特征维度配置

论文位置：

- 消融实验章节

输出重点：

- 各组件贡献对比图
- 不同插件配置的一致性结果

### 第三优先级

然后跑：

```bash
python run_experiment_suite.py --suite thesis_heterogeneity
```

用途：

- 证明方法在更真实的数据异构组合下仍然稳定

论文位置：

- 异构性分析章节
- 真实性实验章节

输出重点：

- 不同异构来源组合下的最终精度

### 第四优先级

接着跑：

```bash
python run_experiment_suite.py --suite thesis_engineering
```

用途：

- 证明你的实现不仅有效，而且文件夹级插件接入形态稳定
- 特别用于支撑“文件夹级插件接入可用”

论文位置：

- 工程实现章节
- 系统实现与部署价值章节

输出重点：

- `FedAvg`、文件夹级插件默认配置、文件夹级插件工程配置对比结果

### 第五优先级

最后补参数分析：

```bash
python run_experiment_suite.py --suite alpha_sweep
python run_experiment_suite.py --suite lambda_sweep
python run_experiment_suite.py --suite dim_sweep
```

用途：

- 证明关键超参数选择合理

论文位置：

- 参数敏感性分析章节

输出重点：

- 参数变化趋势图

## 2. 建议每天怎么跑

如果你是单机跑实验，建议按这个节奏：

### 第一天

- 跑 `thesis_main`
- 检查所有图是否正常生成
- 检查结果是否有明显异常

### 第二天

- 跑 `thesis_ablation`
- 先确保不同插件配置趋势一致

### 第三天

- 跑 `thesis_heterogeneity`
- 观察异构组合下是否出现异常波动

### 第四天

- 跑 `thesis_engineering`
- 跑 `alpha_sweep`

### 第五天

- 跑 `lambda_sweep`
- 跑 `dim_sweep`
- 整理所有 `suite_summary.csv`

## 3. 每组实验跑完后你要检查什么

### `thesis_main`

检查：

- `FedFed` 是否在大多数 `alpha` 下优于 `FedAvg`
- `alpha` 越小的时候收益是否更明显

### `thesis_ablation`

检查：

- 去掉噪声后性能如何
- 降低蒸馏权重后性能如何
- 不同敏感特征维度版本是否保持合理趋势

### `thesis_heterogeneity`

检查：

- 哪种异构组合最难
- 你的方法是否在复杂组合下仍保持优势

### `thesis_engineering`

检查：

- 文件夹级插件不同工程配置是否结果稳定
- 是否能支撑“工程可交付”的论点

### 参数分析

检查：

- 是否存在明显过大/过小的无效参数区间
- 当前默认值是否处于合理位置
- prototype 蒸馏是否在强异构场景下保持稳定收益

## 4. 论文图表建议对应关系

### 图 1

- `thesis_main` 的测试精度对比曲线

### 图 2

- `thesis_main` 的最终精度柱状图

### 图 3

- `thesis_ablation` 的消融柱状图

### 图 4

- `thesis_heterogeneity` 的异构组合对比图

### 图 5

- `alpha_sweep / lambda_sweep / dim_sweep` 中任选 1 到 2 组参数敏感性曲线

### 表 1

- `thesis_main/suite_summary.csv`

### 表 2

- `thesis_ablation/suite_summary.csv`

### 表 3

- `thesis_engineering/suite_summary.csv`

## 5. 最后整理阶段

所有 suite 跑完后，建议做这三件事：

1. 从每个 `suite_summary.csv` 提取最关键结果
2. 把不必要的中间图删掉，只保留论文要用的图
3. 统一图标题、坐标轴名称、方法命名

## 6. 最后一句建议

如果时间不够，不要平均用力。

优先保证：

1. `thesis_main`
2. `thesis_ablation`
3. `thesis_engineering`

这三组完整，论文就已经能站住。
