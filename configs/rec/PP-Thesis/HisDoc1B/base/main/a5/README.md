# **Experiment A5**

---

## **Dataset**

| Dataset Name | Description                                                                            | Size      | Source      | Training/Eval Size | Vocab Size |
| ------------ | -------------------------------------------------------------------------------------- | --------- | ----------- | ------------------ | ---------- |
| hisdoc1b_5m  | A stratified subsample of the original dataset, balanced with respect to book fraction | 5 million | hisdoc1b_2m | 5.0M / 0.03M       | 19000      |

---

## **Experiments**

### **A5.0.0**

---

- Backbone: PPHGNetv2-B4 with `use_last_conv = True` and `class_expand = 512`
- Neck: SequenceEncoder with `reshape` only
- **Initialize backbone with pretrained `PPOCRv5-server` weights**
- Head: `CTCHead`
- Loss: `CTCLoss` with focal loss enabled
- L2 Regularization: 3.0e-03
- Evaluation:

| Timestep   | Epoch | Accuracy | Character Accuracy | Normalized Edit Distance |
| ---------- | ----- | -------- | ------------------ | ------------------------ |
| **T = 73** | 1     |          |                    |                          |
|            | 2     |          |                    |                          |
|            | 3     |          |                    |                          |
|            | 4     |          |                    |                          |
|            | 5     |          |                    |                          |
| **T = 50** | 1     |          |                    |                          |
|            | 2     |          |                    |                          |
|            | 3     |          |                    |                          |
|            | 4     |          |                    |                          |
|            | 5     |          |                    |                          |
| **T = 40** | 1     |          |                    |                          |
|            | 2     |          |                    |                          |
|            | 3     |          |                    |                          |
|            | 4     |          |                    |                          |
|            | 5     |          |                    |                          |

### **A5.0.1**

---

- Backbone: PPHGNetv2-B4 with `use_last_conv = True` and `class_expand = 512`
- Initialize backbone with pretrained `PPOCRv5-server` weights
- Head: `MultiHead` with `CTCHead` and `NRTRHead`
- Loss: `CTCLoss` with focal loss enabled
- Training with `out_char_num = 40` and `out_avg_kernel_size = [4,2]`
- L2 Regularization: 3.0e-03
- Evaluation:

| Timestep   | Epoch | Accuracy | Character Accuracy | Normalized Edit Distance |
| ---------- | ----- | -------- | ------------------ | ------------------------ |
| **T = 73** | 1     |          |                    |                          |
|            | 2     |          |                    |                          |
|            | 3     |          |                    |                          |
|            | 4     |          |                    |                          |
|            | 5     |          |                    |                          |
| **T = 50** | 1     |          |                    |                          |
|            | 2     |          |                    |                          |
|            | 3     |          |                    |                          |
|            | 4     |          |                    |                          |
|            | 5     |          |                    |                          |
| **T = 40** | 1     |          |                    |                          |
|            | 2     |          |                    |                          |
|            | 3     |          |                    |                          |
|            | 4     |          |                    |                          |
|            | 5     |          |                    |                          |

### **A5.0.2**

---

- Backbone: PPHGNetv2-B4 with `use_last_conv = True` and `class_expand = 512`
- Neck: SequenceEncoder with `reshape` only
- Train from scratch
- Head: `CTCHead`
- Loss: `CTCLoss` with focal loss enabled
- Goal: Training with `out_char_num = 40` and `out_avg_kernel_size = [4,2]` to verify whether the model performs better than `out_char_num = 50` on evaluation set.
- Evaluation:

| Timestep   | Epoch | Accuracy | Character Accuracy | Normalized Edit Distance |
| ---------- | ----- | -------- | ------------------ | ------------------------ |
| **T = 73** | 1     |          |                    |                          |
|            | 2     |          |                    |                          |
|            | 3     |          |                    |                          |
|            | 4     |          |                    |                          |
|            | 5     |          |                    |                          |
| **T = 50** |       |          |                    |                          |
|            | 1     |          |                    |                          |
|            | 2     |          |                    |                          |
|            | 3     |          |                    |                          |
|            | 4     |          |                    |                          |
|            | 5     |          |                    |                          |
| **T = 40** |       |          |                    |                          |
|            | 2     |          |                    |                          |
|            | 3     |          |                    |                          |
