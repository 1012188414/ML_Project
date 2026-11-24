# 眼底照片质量下降时，临床信息能否补偿？——BRSET 糖尿病视网膜病变检测

在 BRSET 数据集的糖尿病患者中，用眼底照片检测是否有糖尿病视网膜病变（DR），研究照片质量变差（模糊、降采样等）时，
5 个临床特征（年龄、性别、糖尿病病程、胰岛素使用、高血压记录）能否弥补丢失的图像信息。

| 研究问题 | 对应实验 |
|---|---|
| RQ1 临床信息能否提升 DR 检测？提升是否随照片质量下降而变大？ | A（只用临床特征）、B（标准训练 + 两种融合） |
| RQ2 临床信息应该怎么融合？哪些特征在起作用？ | B（两阶段 vs 端到端；单特征、置换重要性、打乱对照） |
| RQ3 图像模型变鲁棒之后，还需要临床信息吗？ | C1（基础增强）、C2（加降质增强） |

实验 D（感知质量的融合）的代码和结果保留在项目里，但不写进报告。

**主要结果**（测试集 AP，5 个骨干平均，方括号为置信区间）：
- 两阶段融合的临床增益：清晰照片 +0.039 [0.016, 0.067]，降质照片 +0.051 [0.017, 0.087]；增益主要来自糖尿病病程。
- P1（增益随降质变大）+0.012 [-0.011, 0.035]，不显著；P2（两阶段优于端到端）+0.042 [0.006, 0.084]。
- 鲁棒训练与临床信息可以叠加：降质照片 AP 从 0.755（标准训练、只用图像）升到 0.856（C2 + 临床）。
- 补充分析（`results/extra_analysis/`）：照片越差，临床项在融合分数里占比越大；临床信息主要帮助轻度病例，鲁棒训练主要帮助中度病例。

## 文件夹结构

```
DR_experiments_v2/
├── run_all.py               唯一入口：跑全部实验，也可按阶段运行（默认 5 个骨干）
├── README.md                本文件
├── requirements.txt
├── src/                     实验流水线（实验设置全部在 src/common.py）
│   ├── dr_paths.py            所有文件夹位置（不参与代码指纹）
│   ├── common.py              设置：划分、训练协议、降质、实验定义、代码指纹
│   ├── check_data.py  build_cache.py  degrade.py  nets.py  train.py
│   ├── extract_features.py  fit_fusion.py  predict_endtoend.py
│   ├── evaluate_all.py        评估：完整性检查、bootstrap、汇总表、图
│   └── analysis_extra.py      补充分析（补偿机制、互补性；不训练网络）-> results/extra_analysis/
├── data/
│   ├── brset_diabetic.csv  brset_diabetic_inadequate.csv  dr_grade.csv
│   ├── images/                全部 2476 张照片
│   └── cache/                 缩放后的照片、测试集降质版本、清晰度（可由 cache 阶段重建）
├── weights/
│   ├── pretrained/            骨干的 ImageNet 预训练权重（Hugging Face 缓存，离线使用）
│   └── trained/main/          140 个训练好的网络（7 个模型组 × 5 个骨干 × 4 折）：
│                              <模型组>/<骨干>/fold<k>/best.pt + metrics.json
└── results/
    ├── main/                  5 个骨干的全部结果：features/ predictions/ summary/ logs/ signature.json
    └── extra_analysis/        补充分析
```

## 环境

Python 3.10。先按显卡和驱动安装带 CUDA 的 PyTorch（见 pytorch.org），再安装其余依赖：

```bash
pip install -r requirements.txt
```

结果实际使用的软件版本记录在 `results/<运行>/signature.json`。

## 运行

所有命令都在项目根目录执行。

| 目的 | 命令 |
|---|---|
| 跑全部实验（5 个骨干；最后做测试集评估） | `python run_all.py` |
| 只用已有预测重算最终评估（5 个骨干） | `python run_all.py --stages evaluate` |
| 只评估其中几个骨干，例如原来的 4 个 | `python run_all.py --stages evaluate --backbones convnextv2_nano densenet169 efficientnet_b3 mobilenetv3_large_150d` |
| 补充分析 | `python run_all.py --stages extra` |
| 几分钟的冒烟测试（写到 results/main_smoke/） | `python run_all.py --smoke` |

- **可断点续跑**：已经训练好的网络、特征和预测会被跳过，中断后重新运行同一条命令即可。现在 5 个骨干都已完成，
  所以上面的命令只会重新做评估，不会重新训练。
- **只评估部分骨干**会覆盖 `results/main/summary/`；需要 5 骨干的结果时再运行一次不带 `--backbones` 的命令。
- **分阶段运行**：`--stages check cache B C D evaluate`，各阶段的含义见 `run_all.py` 开头的说明。
- **测试集**：测试集上的指标只在最终评估时计算，从不用于选模型或修改方案；分阶段运行时只做验证集评估。
- 常用参数：`--device cuda`，`--workers 0`（Windows 多进程出问题时）。

## 结果在哪里

| 内容 | 位置 |
|---|---|
| 所有汇总指标（带置信区间，按研究层次排列） | `results/main/summary/key_results.csv` |
| 各骨干的单独数值 | `results/main/summary/key_results_by_backbone.csv` |
| 报告用的表 | `results/main/summary/table_*.csv` |
| 图 | `results/main/summary/figures/`，补充分析的图在 `results/extra_analysis/` |
| 每个模型、每种条件的全部指标 | `results/main/summary/metrics.csv` |
| 训练记录（最佳 epoch、验证集 AP、用时） | `results/main/summary/training.csv` |
| 每张照片的预测概率 | `results/main/predictions/` |

## 注意事项

- **代码指纹**：`run_all.py` 会核对 `src/common.py`、`degrade.py`、`nets.py`、`train.py`、`build_cache.py` 和数据 CSV
  的指纹是否与 `results/main/signature.json` 一致；改了实验设置或训练代码就必须用 `--force` 重新训练。
  文件夹位置放在不参与指纹的 `src/dr_paths.py` 里，移动文件夹不会触发重新训练。
  整理文件夹时更新过指纹，原因、旧值和训练时的软件版本见 `results/main/SIGNATURE_NOTE.md`。
- **两批训练**：4 个主骨干先训练（torch 2.10），PVT-v2 后来用完全相同的协议单独训练（torch 2.14），之后合并到同一个运行；
  报告里应写明 PVT-v2 是后来补充的。
- **PVT-v2 的预训练权重**不在 `weights/pretrained/` 里（当时下载到了系统默认的 Hugging Face 缓存）；
  只有重新训练 PVT 时才需要，联网会自动下载。
