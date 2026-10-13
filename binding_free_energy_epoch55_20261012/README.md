# 3DKDavis 结合自由能等效值绘图

本交付仅新增独立推理、绘图脚本和结果文件，没有修改任何现有亲和力模型、训练、数据加载或图结构代码。
按照 karpathy-guidelines 的最小修改原则，复用现有只读模型/图构建接口，不启动训练、不重新构建或写入缓存。

## 模型与评估集

- checkpoint：`/home/dell/chh/output/3dkdavis/jimei/3dkdavis-affinity-axis-geometry-v1_new_kinase/MMCLKin_3dkdavis_1791802853/best_fusion_mse.pkl`
- checkpoint epoch：55；SHA256：`1157648163bf3de07cb6cc4a3cf8c52fd068392ccd92825600b086820d30ba02`
- 实际评估划分：`new_kinase` 留出验证集，曾用于 checkpoint 选择，**不是额外独立测试集**。
- 全部 4,964 个样本均有效，涵盖 68 种药物和 73 个留出激酶；训练 support 为 21,148 对。
- Base/Fusion 来自同一个 checkpoint 的两路输出，并使用完全相同的样本和批次分组。
- 单卡只读推理依次重放原五卡的局部 batch 分组，全局 batch=128。这样保留序列 padding 与原验证的一致性。
- Base 参数冻结，support affinity 边仅来自训练集；验证 affinity 标签没有进入 support 图。

## 标签审计与换算

已用公开 Davis 数据按药物 PubChem ID (`com_name`) 和激酶 `target_name` 对齐，核验本地 CSV 的全部 26,112 条标签：

```text
pKd = 9 - log10(Kd[nM])
最大绝对标签差异 = 4.067e-9
```

参考来源与其内容哈希记录在 `metrics_and_provenance.json` 的 `label_source_audit`。
公开原始数值矩阵、药物与激酶顺序分别来自：

- https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/Y
- https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/ligands_can.txt
- https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/proteins.txt

原始 CSV `affinity` 在 `process_3dkdavis.py` 中原样写入 `.pt` 的 `com_affinity`，
`dataset_loader.py` 再原样作为回归目标 `y`。本次逐个核对全部评估样本的 `.pt` 标签与已验证的 CSV 值。
没有回归目标标准化，因此不需要反标准化。checkpoint 的 affinity normalization 元数据属于图 support 边，
**不是回归目标的标准化**，没有用于改变预测数值。

参考温度和标准浓度：

```text
T = 298.15 K
C° = 1 mol/L
R = 0.00198720425864083 kcal/(mol K)
a = R*T*ln(10) = 1.3642470130342383 kcal/mol
ΔG_equiv = -a*pKd
```

实验轴、预测轴都由 pKd 使用同一线性公式换算；没有再次对 pKd 取对数。
该图表示实验与预测 pKd 在参考温度下换算的结合自由能等效值，
**不是 FEP、MM/PBSA 或新增分子动力学自由能计算**。原实验温度是否为 25°C 未得到确认。

评估集中 `pKd=5` 的记录为 **3,679/4,964**。全数据集中 18,732 条 pKd=5 记录
对应公开 Davis 数值矩阵中的 Kd=10,000 nM。这解释了实验轴约 **−6.821235 kcal/mol** 的水平堆积。
现有数值文件不包含原实验的逐条测量限定符，不能把全部这些记录一概认定为精确测量或删失值。
没有删除这些样本；相关解释应保留这一信息缺口。

## 指标

全部指标使用完整评估集；均值误差定义为 `prediction - experimental`。

| 输出及尺度 | MAE | MSE | RMSE | R² | Pearson | Spearman | CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fusion / pKd | 0.322972 | 0.274853 | 0.524265 | 0.555717 | 0.748733 | 0.599026 | 0.854156 |
| Fusion / kcal/mol | 0.440613 | 0.511549 | 0.715226 | 0.555717 | 0.748733 | 0.599026 | 0.854156 |
| Base / pKd | 0.334898 | 0.293091 | 0.541379 | 0.526237 | 0.739297 | 0.578669 | 0.841752 |
| Base / kcal/mol | 0.456884 | 0.545492 | 0.738574 | 0.526237 | 0.739297 | 0.578669 | 0.841752 |

自由能尺度的 MSE 单位是 `(kcal/mol)²`；MAE、RMSE、均值误差单位是 `kcal/mol`。

| 容限 | Fusion 样本数 / 比例 | Base 样本数 / 比例 |
|---|---:|---:|
| 绝对误差 ≤ 0.5 kcal/mol | 3,623 / 72.99% | 3,667 / 73.87% |
| 绝对误差 ≤ 1.0 kcal/mol | 4,356 / 87.75% | 4,326 / 87.15% |

Fusion 自由能均值误差为 **+0.020975 kcal/mol**；Base 为 **+0.125008 kcal/mol**。
Fusion 整体 MAE/RMSE 改善，但 ≤0.5 kcal/mol 的样本数比 Base 少 44；不能声称全部误差容限均改善。

已验证：`MAE_ΔG=a*MAE_pKd`、`RMSE_ΔG=a*RMSE_pKd`、`MSE_ΔG=a²*MSE_pKd`，
以及 Pearson、Spearman、CI、R² 换算前后一致。
复现的 pKd 指标与训练日志的最大差异约 1.9e-7，属于数值精度差异。
常数标签/预测时无法定义的相关系数在脚本中返回 `null`，不会用 0 冒充。

## 图与文件

- `plots/free_energy_consistency.*`：Fusion 全样本主图。
- `plots/free_energy_dense_zoom.*`：联合数据 1%～99% 分位数窗口，窗口内 4,782 条、窗口外 182 条。
- `plots/free_energy_error_distribution.*`：完整误差范围的直方图，标出 0、±1 kcal/mol。
- `plots/free_energy_base_vs_fusion.*`：同 checkpoint 的 Base/Fusion，坐标、hexbin 网格与颜色尺度一致。
- `predictions.csv`：逐样本 pKd、自由能、误差、药物与激酶 ID，附 Base 对应列。
- `metrics.csv`：两种尺度、两路输出的全部指标；容限列的单位由对应 scale 决定。
- `metrics_and_provenance.json`：推理配置、标签审计、checkpoint/源码哈希、指标一致性检查。
- `plots/plot_metrics.json`：绘图配置、实际坐标范围、窗口范围和样本数。

由于 pKd=5 大量重叠，主图与比较图采用 hexbin，颜色表示每格样本数并使用对数归一化。
脚本检查主图和比较图各面板的 hexbin 计数和等于 4,964；没有随机抽样、删点或坐标抖动。
两轴使用联合数据范围加 5% 余量，实际主图范围约为 `[-14.604734, -5.708494]`，等比例正方形。
黑线为 y=x；浅灰虚线是 ±1 kcal/mol **误差参考线，不是置信区间**。
橙色框标出分位数窗口；局部图中的指标仍由完整 4,964 条计算。

全部图提供 **600 dpi PNG、PDF 和 SVG**。SVG 保留文本，PDF 嵌入 TrueType 字体，hexbin 使用向量图元。

建议主图图注：

> Predicted versus experimental equivalent binding free energies obtained by converting pKd at reference T=298.15 K and C°=1 mol/L. Both axes use ΔG_equiv=−RT ln(10) pKd. All 4,964 new-kinase validation pairs are included; hexbin colors show log-scaled sample counts. The solid line indicates y=x and dashed lines mark ±1 kcal/mol error references, not confidence intervals. This is a pKd-based equivalent-value conversion, not FEP, MM/PBSA, or a new molecular-dynamics free-energy calculation; the original experimental temperature is unconfirmed. This validation set was used for checkpoint selection.

## 运行命令

仅重新绘图，不加载模型、无需联网：

```bash
cd /home/dell/chh/fl-MMCLKIN-GNN
conda run --no-capture-output -n MMCLKin python plot_binding_free_energy.py \
  --csv binding_free_energy_epoch55_20261012/predictions.csv \
  --output_dir binding_free_energy_epoch55_20261012/plots
```

重新推理（新增目录，现有结果不会覆盖；联网核验公开 Davis 源标签）：

```bash
cd /home/dell/chh/fl-MMCLKIN-GNN
conda run --no-capture-output -n MMCLKin python -u evaluate_binding_free_energy.py \
  --checkpoint /home/dell/chh/output/3dkdavis/jimei/3dkdavis-affinity-axis-geometry-v1_new_kinase/MMCLKin_3dkdavis_1791802853/best_fusion_mse.pkl \
  --output_dir binding_free_energy_epoch55_recheck \
  --device cuda:0 --evaluation_world_size 5
```

独立脚本位于项目根目录：`evaluate_binding_free_energy.py` 和 `plot_binding_free_energy.py`。
