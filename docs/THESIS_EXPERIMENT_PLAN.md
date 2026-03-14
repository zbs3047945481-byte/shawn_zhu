# 毕设实验计划

这份文档给出当前项目已经固化好的论文实验套件，以及推荐执行顺序。

## 1. 推荐实验顺序

### 第一组：主结果

```bash
python run_experiment_suite.py --suite thesis_main
```

用途：

- 比较 `FedAvg` 与 `fedfed_prototype`
- 在不同 `dirichlet_alpha` 下验证方法有效性

建议放在论文：

- 主要对比实验章节

### 第二组：消融实验

```bash
python run_experiment_suite.py --suite thesis_ablation
```

用途：

- 对比无插件、标准 prototype 插件、去噪版、弱蒸馏版、单文件版
- 验证各组件作用

建议放在论文：

- 消融实验章节

### 第三组：异构性组合实验

```bash
python run_experiment_suite.py --suite thesis_heterogeneity
```

用途：

- 比较不同异构来源组合下的方法稳定性

建议放在论文：

- 真实性实验或异构性分析章节

### 第四组：工程一致性实验

```bash
python run_experiment_suite.py --suite thesis_engineering
```

用途：

- 比较 `FedAvg`
- 比较标准插件版本
- 比较单文件交付版本

建议放在论文：

- 工程实现与部署价值章节

### 第五组：超参数敏感性实验

```bash
python run_experiment_suite.py --suite lambda_sweep
python run_experiment_suite.py --suite dim_sweep
python run_experiment_suite.py --suite alpha_sweep
```

用途：

- 观察关键超参数变化趋势

建议放在论文：

- 参数敏感性分析章节

## 2. 当前套件列表

当前 `run_experiment_suite.py` 支持：

- `baseline_vs_plugin`
- `alpha_sweep`
- `lambda_sweep`
- `dim_sweep`
- `thesis_main`
- `thesis_ablation`
- `thesis_heterogeneity`
- `thesis_engineering`

## 3. 每个 suite 的输出

每个实验套件目录下会生成：

- 对比曲线图
- 柱状图
- `suite_summary.json`
- `suite_summary.csv`

适合直接用于：

- 论文图表整理
- 实验结果总表整理
- 后续手工筛选最佳配置

## 4. 执行清单

推荐直接配合：

- `docs/THESIS_EXECUTION_CHECKLIST.md`

这份清单已经按优先级写好了“先跑什么、后跑什么、结果放哪里”。
