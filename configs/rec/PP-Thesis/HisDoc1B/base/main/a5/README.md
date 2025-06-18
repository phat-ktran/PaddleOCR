# Experiment A5: OCR Model Performance Analysis

## Executive Summary

- **Objective**: Evaluate PPHGNetv2-B4 backbone configurations for OCR tasks
- **Key Finding**: Models consistently overfit after the first epoch, 32x640 image size gives better accuracy than 32x584 one
- **Best Configuration**: A5.0.0.2 achieved 69.85% accuracy before overfitting

## Dataset Information

| Attribute       | Value                                           |
| --------------- | ----------------------------------------------- |
| **Name**        | hisdoc1b_5m                                     |
| **Description** | Stratified subsample, balanced by book fraction |
| **Size**        | 5 million samples                               |
| **Source**      | hisdoc1b_2m                                     |
| **Split**       | 5.0M training / 0.06M eval                      |
| **Vocabulary**  | 19,000 characters                               |

## Experiment Overview

### Configuration Matrix

| Experiment | Backbone Config              | Image Size | Neck            | Head      | Key Changes                                      | Model Size |
| ---------- | ---------------------------- | ---------- | --------------- | --------- | ------------------------------------------------ | ---------- |
| A5.0.0     | class_expand=384             | 32×584     | SequenceEncoder | CTC       | AMP training, Baseline with pretrained weights   | 21,700,889 |
| A5.0.0.1   | class_expand=256             | 32×640     | SequenceEncoder | CTC       | AMP training, Reduced class_expand, larger width | 19,006,617 |
| A5.0.0.2   | class_expand=384             | 32×640     | SequenceEncoder | CTC       | AMP training, Larger width                       | 21,700,889 |
| A5.0.1     | Default + SVTR=160, NRTR=512 | 32x584     | None            | MultiHead | Multi-head approach, AMP training                | 22,212,009 |
| A5.0.1.2   | Default PPOCRv5              | 32×640     | None            | MultiHead | Full precision training                          | 21,169,089 |
| A5.1.0     | SVTRNet                      | 32×640     | None            | CTC       | AMP training, no pretrained weights              | 22,383,545 |
| A5.1.0.1   | SVTRNet                      | 32×640     | None            | MultiHead | Full precision training                          | 21,169,089 |
| A5.2.0     | Default PPOCRv4              | 32×640     | None            | MultiHead | Full precision training                          | 21,169,089 |

### Evaluation Set Performance Summary

| Experiment | Best Checkpoint | Accuracy | Character Accuracy | Normalized Edit Distance | Notes                                          |
| ---------- | --------------- | -------- | ------------------ | ------------------------ | ---------------------------------------------- |
| A5.0.0     | Epoch 1         | 69.75%   | 94.17%             | 94.88%                   | PPHGNetv2-B4 + CTCHead + 32x584                |
| A5.0.0.1   | Epoch 1         | 67.63%   | 93.56%             | 94.39%                   | PPHGNetv2-B4 + CTCHead + 32x640 + Smaller Neck |
| A5.0.0.2   | Epoch 1         | 69.85%   | 95.00%             | 94.82%                   | PPHGNetv2-B4 + CTCHead + 32x640                |
| A5.0.1.2   | Epoch 1         | 63.38%   | 91.93%             | 93.57%                   | PPHGNetv2-B4 + MultiHead                       |
| A5.1.0     | Epoch 14        | 77.14%   | 96.48%             | 96.10%                   | SVTRNet + CTCHead                              |
| A5.2.0     | Epoch 8         | 71.79%   | 95.34%             | 94.98%                   | PPHGNet-small + MultiHead                      |

## Detailed Results

### A5.0.0: Baseline Configuration

**Setup:**

- Backbone: PPHGNetv2-B4 (`use_last_conv=True`, `class_expand=384`)
- Image: 32×584, Neck: SequenceEncoder, Head: CTCHead
- Regularization: L2 3.0e-05
- Link to report: [Report](https://wandb.ai/trankim147-vnu-hcmus/HisDoc1B-5M/reports/Finetuning-PPOCRv5--VmlldzoxMzE3MzMzMg?accessToken=cuhjhoog5hi2bm9gh3hotlin5979tg867rxo8qxxmdfqu24y3qtf5dhexqgnykhg)

**Outcome:** Training halted after 5 epochs due to overfitting

### A5.0.0.1: Customized Configuration

**Setup:**

- Modified: `class_expand=256`, image size 32×640
- Same training protocol as A5.0.0
- Regularization: L2 3.0e-03 for epoch 1; 3.0e-05 from epoch 2
- Link to report: [Report](https://wandb.ai/trankim147-vnu-hcmus/HisDoc1B-5M/reports/Finetuning-PPOCRv5--VmlldzoxMzE3MzMzMg?accessToken=cuhjhoog5hi2bm9gh3hotlin5979tg867rxo8qxxmdfqu24y3qtf5dhexqgnykhg)

**Analysis:** Sharp performance drop after 4000 additional steps indicates severe overfitting

### A5.0.0.2: Customized Configuration

**Setup:**

- Modified: `class_expand=384`, image size 32×640
- Same training protocol as A5.0.0
- Regularization: L2 3.0e-03
- Link to report: [Report](https://wandb.ai/trankim147-vnu-hcmus/HisDoc1B-5M/reports/Finetuning-PPOCRv5--VmlldzoxMzE3MzMzMg?accessToken=cuhjhoog5hi2bm9gh3hotlin5979tg867rxo8qxxmdfqu24y3qtf5dhexqgnykhg)

### A5.0.1: Multi-Head Approach

**Setup:**

- Enhanced dimensions: SVTR neck 120→160, NRTR head 384→512
- MultiHead: CTCHead + NRTRHead
- AMP training with dynamic loss scaling
- Link to report: [Report](https://api.wandb.ai/links/trankim147-vnu-hcmus/exts5hb7)

**Issues:**

- Loss instability from step 6000
- Training discontinued due to convergence problems

### A5.0.1.2: Stable Multi-Head

**Setup:**

- Default PPOCRv5 configurations
- Full precision training (no AMP)
- Validation every 3000 steps after epoch 1
- Link to report: [Report](https://wandb.ai/trankim147-vnu-hcmus/HisDoc1B-5M/reports/Finetuning-PPOCRv5-Part-3--VmlldzoxMzE3MzM4Mg?accessToken=oi8h83yigkxon6xpov6o2o6af84gcmxwf4s2xegl6amdfw4u5cfhghgma1cinmzd)

**Status:** Training halted at epoch 1 + 3000 steps due to overfitting

### A5.1.0: Baseline Configuration

**Setup:**

- Backbone: SVTRNet
- Image: 32×640, Neck: SequenceEncoder, Head: CTCHead
- Regularization: `dropout=0.2` `drop_path=0.1` `attn_dropout=0.1`
- Link to report: [Report](https://api.wandb.ai/links/trankim147-vnu-hcmus/1zxkjie2)

**Outcome:** Still training

### A5.2.0: Baseline Configuration

**Setup:**

- Backbone: PPHGNet
- Image: 32×640, Neck: None, Head: MultiHead
- Regularization: L2 3.0e-05
- Link to report: [Report](https://wandb.ai/trankim147-vnu-hcmus/HisDoc1B-5M/reports/PPOCRv4--VmlldzoxMzE3MzI2Mg?accessToken=7sertx09hmz6j87d3ety9vakbjhit70f2qb91ce99s5bkqwm570lesq7j0j4bedf)

**Outcome:** Still training

## Key Findings & Patterns

### Critical Issues

1. **Consistent Overfitting**: All experiments show degradation after epoch 1
2. **AMP Instability**: Mixed precision caused training issues in A5.0.1

### Insights

1. **Optimal Configuration**: A5.0.0.2 shows best performance before overfitting
2. **Regularization Insufficient**: Current L2 values don't prevent overfitting
