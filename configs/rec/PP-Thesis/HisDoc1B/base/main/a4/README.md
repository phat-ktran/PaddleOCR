# Experiment A4

## Dataset:

- paddle_v5_ss1 (hisdoc1b_2m)
- hisdoc1b_10m
- hisdoc1b_20m
- hisdoc1b_30m

## Goals:

- Pretrain models with incremental datasets

## Experiment list

### A4.0

- Backbone: PPHGNetv2-B4 with `use_last_conv = True` and `class_expand = 512`
- Neck: SequenceEncoder with `reshape` only
- Head: `CTCHead`
- Loss: `CTCLoss` with focal loss enabled

### A4.1

- Backbone: PPHGNetv2-B4 with `use_last_conv = True` and `class_expand = 512`
- Neck: SequenceEncoder with `reshape` only
- Head: `MultiHead` of `CTCHead` and `NRTRHead`
- Loss: `MultiLoss` of `CTCLoss` with focal loss enabled and `NRTRLoss`

### A4.2

- Backbone: PPHGNetv2-B4 
- Head: `MultiHead` of `CTCHead` and `NRTRHead` where `CTCHead` has neck of 2 layers of `SVTR` blocks
- Loss: `MultiLoss` of `CTCLoss` with focal loss enabled and `NRTRLoss`
- Dataset: `MultiScaleLMDBDataSet`

This is the original settings of PPOCRv5 model.