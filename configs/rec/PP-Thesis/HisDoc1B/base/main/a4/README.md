# **Experiment A4**

---

## **Dataset**

| Dataset Name  | Description                                                                            | Size      | Source      | Training/Eval Size | Vocab Size |
| ------------- | -------------------------------------------------------------------------------------- | --------- | ----------- | ------------------ | ---------- |
| paddle_v5_ss1 | A stratified subsample of the original dataset, balanced with respect to book fraction | 2 million | hisdoc1b_2m | 1.6M / 0.4M        | 25869      |

---

## **Experiments**

### **A4.0** & **A4.0.0**

---

- Backbone: PPHGNetv2-B4 with `use_last_conv = True` and `class_expand = 512`
- Neck: SequenceEncoder with `reshape` only
- Initialize backbone with pretrained `PPOCRv5-server` weights
- Head: `CTCHead`
- Loss: `CTCLoss` with focal loss enabled
- Goal: A simple run to verify the correctness of the model settings. Specifically, we need to check whether `out_char_num = 50` and `out_avg_kernel_size = [4,2]` gain optimal performance.
- Evaluation:

#### L2 Regularization: 1.0e-06

| Timestep   | Epoch | Accuracy   | Character Accuracy | Normalized Edit Distance |
| ---------- | ----- | ---------- | ------------------ | ------------------------ |
| **T = 74** | 1     | 0.6106     | 0.8548             | 0.9339                   |
|            | **2** | **0.6288** | **0.8547**         | **0.9402**               |
|            | 3     | 0.5948     | 0.8302             | 0.9336                   |
|            | 4     | 0.5114     | 0.7520             | 0.9188                   |
|            | 5     | 0.3845     | 0.7005             | 0.8754                   |
| **T = 50** | 1     | 0.6629     | 0.9473             | 0.9444                   |
|            | **2** | **0.6981** | **0.9560**         | **0.9522**               |
|            | 3     | 0.6841     | 0.9394             | 0.9495                   |
|            | 4     | 0.6065     | 0.8627             | 0.9350                   |
|            | 5     | 0.3399     | 0.5693             | 0.8594                   |
| **T = 40** | **2** | **0.2993** | **0.4631**         | **0.8519**               |
|            | 3     | 0.2011     | 0.2787             | 0.7688                   |

#### L2 Regularization: 1.0e-02

| Timestep   | Epoch | Accuracy | Character Accuracy | Normalized Edit Distance |
| ---------- | ----- | -------- | ------------------ | ------------------------ |
| **T = 74** | 1     |          |                    |                          |
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

### **A4.0.1**

---

- Backbone: PPHGNetv2-B4 with `use_last_conv = True` and `class_expand = 512`
- Neck: SequenceEncoder with `reshape` only
- Initialize backbone with pretrained `PPOCRv5-server` weights
- Head: `CTCHead`
- Loss: `CTCLoss` with focal loss enabled
- Goal: Training with `out_char_num = 40` and `out_avg_kernel_size = [4,2]` to verify whether the model performs better than `out_char_num = 50` on evaluation set.
- Evaluation:

#### L2 Regularization: 1.0e-02

| Timestep   | Epoch | Accuracy | Character Accuracy | Normalized Edit Distance |
| ---------- | ----- | -------- | ------------------ | ------------------------ |
| **T = 74** | 1     |          |                    |                          |
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

### **A4.0.2**

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
| **T = 74** |       |          |                    |                          |
|            | 1     |          |                    |                          |
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
