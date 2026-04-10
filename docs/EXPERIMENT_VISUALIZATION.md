# 实验结果可视化说明

当前项目已经支持两类实验图表生成：

## 1. 单次实验自动出图

每次训练结束后，程序会在对应实验目录下自动生成：

- `metrics.json`
- `test_acc_curve.png`
- `test_loss_curve.png`

这些文件默认位于：

```text
result/<dataset_name>/<exp_name>/
```

其中：

- `metrics.json` 保存完整指标
- `test_acc_curve.png` 保存测试精度曲线
- `test_loss_curve.png` 保存测试损失曲线

## 2. 多实验对比图

项目新增了：

- `plot_experiments.py`
- `run_experiment_suite.py`

它可以读取多个 `metrics.json`，生成：

- 测试精度对比曲线
- 测试损失对比曲线
- 最佳精度柱状图
- 最终精度柱状图

示例：

```bash
python plot_experiments.py \
  --metrics \
    result/mnist/exp_a/metrics.json \
    result/mnist/exp_b/metrics.json \
  --labels \
    FedAvg \
    FedFed \
  --output_dir result/comparisons
```

如果你希望批量执行常见实验并自动生成对比图，可以直接使用：

```bash
python run_experiment_suite.py --suite baseline_vs_plugin
```

当前内置实验套件包括：

- `baseline_vs_plugin`
- `alpha_sweep`
- `lambda_sweep`
- `dim_sweep`
- `thesis_main`
- `thesis_ablation`
- `thesis_heterogeneity`
- `thesis_engineering`

## 3. 适合论文的常见对比组合

你后续可以按这些组合直接出图：

- `FedAvg` vs `fedfed_prototype`
- 不同 `dirichlet_alpha`
- 不同 `fedfed_lambda_distill`
- 不同 `fedfed_sensitive_dim`
- 不同 `fedfed_noise_sigma`
- 不同异构强度和不同蒸馏强度配置

## 4. 当前可视化相关代码位置

- 自动单次实验出图：
  - `src/utils/metrics.py`
  - `src/utils/plotting.py`

- 多实验对比图：
  - `plot_experiments.py`
  - `run_experiment_suite.py`

## 5. 依赖

绘图依赖：

- `matplotlib`

如果环境中没有安装，训练仍可进行，但图表不会生成。

## 6. 论文实验建议

推荐直接查看：

- `docs/THESIS_EXPERIMENT_PLAN.md`

里面已经把当前内置 suite 和论文实验章节做了对应。

## 7. 多随机种子汇总

当 `run_experiment_suite.py` 使用：

```bash
--num_repeats 3
```

时，每个配置会重复运行多个随机种子，并额外输出：

- `suite_summary_multiseed.csv`

适合直接用于论文中的 `mean ± std` 结果表。
