# DALI Example Configurations

This directory contains example configuration files demonstrating the use of NVIDIA DALI operators in PaddleOCR.

## Available DALI Configurations

### Text Detection
- **det_r50_vd_db_dali.yml**: Text detection using ResNet50_vd backbone with DALI-accelerated data preprocessing

### Text Recognition
- **rec_svtr_dali.yml**: Text recognition using SVTR model with DALI-accelerated data preprocessing

## Usage

To use these configurations, you need to:

1. **Install NVIDIA DALI** (optional):
   ```bash
   # For CUDA 11.x
   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
   
   # For CUDA 12.x
   pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120
   ```

2. **Train with DALI config**:
   ```bash
   # Text Detection
   python tools/train.py -c configs/det/det_r50_vd_db_dali.yml
   
   # Text Recognition
   python tools/train.py -c configs/rec/rec_svtr_dali.yml
   ```

## DALI Operators in Configs

The DALI configurations use the following operators:

- **DALIDecodeImage**: GPU-accelerated image decoding
- **DALINormalizeImage**: GPU-accelerated normalization
- **DALIRandomFlip**: Random horizontal/vertical flip
- **DALIRandomRotation**: Random rotation within angle range
- **DALIBrightness**: Random brightness adjustment
- **DALIContrast**: Random contrast adjustment

## Customization

You can customize DALI operators in the configuration files:

```yaml
transforms:
  - DALIDecodeImage:
      img_mode: BGR
      device: cpu  # Change to 'gpu' for GPU acceleration
  - DALIRandomFlip:
      horizontal: true
      vertical: false
      probability: 0.5  # Adjust probability
  - DALIBrightness:
      brightness_range: [0.8, 1.2]  # Adjust range
      probability: 0.3
```

## Performance Tips

1. **Device Selection**: Set `device: gpu` in DALI operators when you have spare GPU memory
2. **Batch Size**: Larger batch sizes benefit more from DALI acceleration
3. **Num Workers**: Adjust `num_workers` based on your CPU/GPU resources
4. **Mixed Pipeline**: You can mix DALI operators with standard PaddleOCR operators

## Documentation

For more information, see:
- [DALI Integration Documentation](../../doc/dali_integration.md)
- [NVIDIA DALI Documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)

## Note

DALI is optional. If not installed, you can use the standard PaddleOCR configurations instead.
