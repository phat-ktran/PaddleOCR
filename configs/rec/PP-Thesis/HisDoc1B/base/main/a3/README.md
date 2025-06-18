# Experiment A3

## Dataset: hisdoc1b_greedy_41k

## Goals:

- Verify whether the model can learn with a large vocabulary of 31290 characters.

## Experiment list

### A3.0

- Backbone: PPHGNetv2-B4 with `use_last_conv = True` and `class_expand = 512`
- Neck: SequenceEncoder with `reshape` only
- Head: `CTCHead`
- Loss: `CTCLoss` with focal loss enabled

### A3.1

- Backbone: PPHGNetv2-B4 with `use_last_conv = True` and `class_expand = 512`
- Neck: SequenceEncoder with `reshape` only
- Head: `MultiHead` of `CTCHead` and `NRTRHead`
- Loss: `MultiLoss` of `CTCLoss` with focal loss enabled and `NRTRLoss`

### A3.2

- Backbone: PPHGNetv2-B4 
- Head: `MultiHead` of `CTCHead` and `NRTRHead` where `CTCHead` has neck of 2 layers of `SVTR` blocks
- Loss: `MultiLoss` of `CTCLoss` with focal loss enabled and `NRTRLoss`
- Dataset: `MultiScaleLMDBDataSet`

This is the original settings of PPOCRv5 model.

### A3.3

- Backbone: SVTRNet with `out_channels = 512`
- Neck: SequenceEncoder with `reshape` only
- Head: `CTCHead`
- Loss: `CTCLoss` with focal loss enabled
