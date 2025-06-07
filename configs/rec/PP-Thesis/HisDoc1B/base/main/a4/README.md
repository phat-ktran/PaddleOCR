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
- GPU: 8 x RTX 3090 24GB
- Runtime: 165 mins
- Goal: A simple run to verify the correctness of the model settings. Specifically, we need to check whether `out_char_num = 50` and `out_avg_kernel_size = [4,2]` gain optimal performance.
- Evaluation:

#### L2 Regularization: 1.0e-06 (RTX 3090 24GB)

| Timestep   | Epoch | Accuracy   | Character Accuracy | Normalized Edit Distance |
| ---------- | ----- | ---------- | ------------------ | ------------------------ |
| **T = 73** | 1     | 0.6106     | 0.8548             | 0.9339                   |
|            | **2** | **0.6288** | **0.8547**         | **0.9402**               |
|            | 3     | 0.5948     | 0.8302             | 0.9336                   |
|            | 4     | 0.5114     | 0.7520             | 0.9188                   |
|            | 5     | 0.3845     | 0.7005             | 0.8754                   |
| **T = 50** | 1     | 0.6629     | 0.9473             | 0.9444                   |
|            | **2** | **0.6981** | **0.9560**         | **0.9522**               |
|            | 3     | 0.6841     | 0.9394             | 0.9495                   |
|            | 4     | 0.6065     | 0.8627             | 0.9350                   |
|            | 5     | 0.3399     | 0.5693             | 0.8594                   |
| **T = 40** | 1     | 0.3172     | 0.4958             | 0.85742                  |
|            | 2     | 0.2993     | 0.4631             | 0.8519                   |
|            | 3     | 0.2011     | 0.2787             | 0.7688                   |

### **A4.0.1**

---

- Backbone: PPHGNetv2-B4 with `use_last_conv = True` and `class_expand = 512`
- Neck: SequenceEncoder with `reshape` only
- Initialize backbone with pretrained `PPOCRv5-server` weights
- Head: `CTCHead`
- Loss: `CTCLoss` with focal loss enabled
- GPU: 8 x RTX 3090 24GB
- Runtime: 165 mins
- Goal: Training with `out_char_num = 40` and `out_avg_kernel_size = [4,2]` to verify whether the model performs better than `out_char_num = 50` on evaluation set.
- Evaluation:

#### L2 Regularization: 1.0e-02 (RTX 3060 12GB & RTX 4070S Ti 16GB)

| Timestep   | Epoch | Accuracy           | Character Accuracy | Normalized Edit Distance |
| ---------- | ----- | ------------------ | ------------------ | ------------------------ |
| **T = 73** | 1     | 0.6149607617979574 | 0.8595995846841524 | 0.935286715701296        |
|            | 2     | 0.6354686206849752 | 0.8766606086911308 | 0.940179808171752        |
|            | 3     | 0.6220283854811952 | 0.8638597356517836 | 0.9364579272371905       |
|            | 4     | 0.428149992614167  | 0.8291178186879805 | 0.8858016723569082       |
|            | 5     | 0.5532021810243378 | 0.8134995558604349 | 0.9211807328163639       |
| **T = 50** | 1     | 0.6514339000769654 | 0.9173871964155644 | 0.9423313310783225       |
|            | 2     | 0.6696042180570758 | 0.9212998123685527 | 0.9466734962778695       |
|            | 3     | 0.6598915480855941 | 0.919442758036071  | 0.9434796998034296       |
|            | 4     | 0.5010512683846704 | 0.8875056646226487 | 0.9043737224099343       |
|            | 5     | 0.6185133239677065 | 0.8955843811071021 | 0.9346468893406915       |
| **T = 40** | 1     | 0.6574990062161713 | 0.9349121894844644 | 0.944014232568019        |
|            | 2     | 0.6811544201853243 | 0.9443445253988322 | 0.948926982672158        |
|            | 3     | 0.6684091971442397 | 0.9379651962317428 | 0.9458545283082695       |
|            | 4     | 0.5582547694445089 | 0.9404527222772292 | 0.9181602729148468       |
|            | 5     | 0.6392361866172849 | 0.930523038231737  | 0.9392707436061829       |

### **A4.0.2**

---

- Backbone: PPHGNetv2-B4 with `use_last_conv = True` and `class_expand = 512`
- Neck: SequenceEncoder with `reshape` only
- Train from scratch
- Head: `CTCHead`
- Loss: `CTCLoss` with focal loss enabled
- Goal: Training with `out_char_num = 40` and `out_avg_kernel_size = [4,2]` from scratch to verify whether the model performs better than those initialized with pretrained weights.
- Evaluation:

| Timestep   | Epoch | Accuracy | Character Accuracy | Normalized Edit Distance |
| ---------- | ----- | -------- | ------------------ | ------------------------ |
| **T = 73** |       |          |                    |                          |
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
|            | 1     |          |                    |                          |
|            | 2     |          |                    |                          |
|            | 3     |          |                    |                          |
|            | 4     |          |                    |                          |
|            | 5     |          |                    |                          |

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
