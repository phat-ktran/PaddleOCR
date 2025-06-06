# **Experiment A5**

---

## **Dataset**

| Dataset Name  | Description                                                                            | Size      | Source      | Training/Eval Size | Vocab Size |
| ------------- | -------------------------------------------------------------------------------------- | --------- | ----------- | ------------------ | ---------- |
| paddle_v5_ss1 | A stratified subsample of the original dataset, balanced with respect to book fraction | 2 million | hisdoc1b_2m | 1.6M / 0.4M        | 25869      |

---

## **Experiments**

### **A5.0** & **A5.0.0**

---

- Backbone: PPHGNetv2-B4 with `use_last_conv = True` and `class_expand = 512`
- Neck: SequenceEncoder with `reshape` only
- Initialize backbone with pretrained `PPOCRv5-server` weights
- Head: `CTCHead`
- Loss: `CTCLoss` with focal loss enabled
- Goal: A simple run to verify the correctness of the model settings. Specifically, we need to check whether `out_char_num = 50` and `out_avg_kernel_size = [4,2]` gain optimal performance.
- Evaluation:

#### L2 Regularization: 1.0e-02

| Timestep   | Epoch | Accuracy | Character Accuracy | Normalized Edit Distance |
| ---------- | ----- | -------- | ------------------ | ------------------------ |
| **T = 73** | 1     |          |                    |                          | |            | 2     |          |                    |                          |
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
- Neck: SequenceEncoder with `reshape` only
- Initialize backbone with pretrained `PPOCRv5-server` weights
- Head: `CTCHead`
- Loss: `CTCLoss` with focal loss enabled
- Goal: Training with `out_char_num = 40` and `out_avg_kernel_size = [4,2]` to verify whether the model performs better than `out_char_num = 50` on evaluation set.
- Evaluation:

#### L2 Regularization: 1.0e-02

| Timestep   | Epoch | Accuracy | Character Accuracy | Normalized Edit Distance |
| ---------- | ----- | -------- | ------------------ | ------------------------ |
| **T = 73** | 1     |          |                    |                          | |            | 2     |          |                    |                          |
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
| **T = 73** |       |          |                    |                          | |            | 1     |          |                    |                          |
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