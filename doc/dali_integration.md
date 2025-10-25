# NVIDIA DALI Integration for PaddleOCR

## Overview

This document describes the integration of NVIDIA DALI (Data Loading Library) with PaddleOCR. DALI is a GPU-accelerated library for data loading and preprocessing that can significantly speed up data processing pipelines, especially when training on GPUs.

## What is NVIDIA DALI?

NVIDIA DALI is a library for data loading and pre-processing to accelerate deep learning applications. It provides a collection of highly optimized building blocks for loading and processing image, video and audio data. DALI can leverage GPU acceleration to speed up data preprocessing, which is particularly beneficial when the data preprocessing becomes a bottleneck.

### Key Benefits

- **GPU Acceleration**: Offload data preprocessing to GPU, freeing up CPU resources
- **Performance**: Significant speedup in data loading and augmentation
- **Pipeline Optimization**: Optimized parallel execution of data operations
- **Memory Efficiency**: Efficient memory management for large-scale training

## Installation

To use DALI operators in PaddleOCR, you need to install NVIDIA DALI. The installation depends on your CUDA version:

### For CUDA 11.x
```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
```

### For CUDA 12.x
```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120
```

### For CPU-only
```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali
```

**Note**: DALI is optional. If not installed, PaddleOCR will fall back to standard operators.

## Available DALI Operators

The following DALI-accelerated operators are available in PaddleOCR:

### 1. DALIDecodeImage
GPU-accelerated image decoding from bytes.

```yaml
- DALIDecodeImage:
    img_mode: BGR  # or 'RGB', 'GRAY'
    device: cpu    # or 'gpu' for GPU acceleration
    output_dtype: uint8
```

### 2. DALINormalizeImage
GPU-accelerated image normalization (scaling, mean subtraction, std division).

```yaml
- DALINormalizeImage:
    scale: 1./255.
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    order: 'hwc'  # or 'chw'
```

### 3. DALIResize
GPU-accelerated image resizing.

```yaml
- DALIResize:
    size: [640, 640]  # [height, width]
    interp_type: 'LINEAR'
```

### 4. DALIRandomRotation
Randomly rotate images within specified angle range.

```yaml
- DALIRandomRotation:
    angle_range: [-10, 10]  # rotation range in degrees
    probability: 0.5
```

### 5. DALIRandomFlip
Randomly flip images horizontally and/or vertically.

```yaml
- DALIRandomFlip:
    horizontal: true
    vertical: false
    probability: 0.5
```

### 6. DALIBrightness
Randomly adjust image brightness.

```yaml
- DALIBrightness:
    brightness_range: [0.8, 1.2]
    probability: 0.5
```

### 7. DALIContrast
Randomly adjust image contrast.

```yaml
- DALIContrast:
    contrast_range: [0.8, 1.2]
    probability: 0.5
```

## Usage Examples

### Example 1: Text Detection with DALI

Replace standard operators in your detection config:

```yaml
Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/icdar2015/text_localization/
    label_file_list:
      - ./train_data/icdar2015/text_localization/train_icdar2015_label.txt
    transforms:
      # Replace DecodeImage with DALIDecodeImage
      - DALIDecodeImage:
          img_mode: BGR
          device: cpu
      - DetLabelEncode:
      # Add DALI augmentation operators
      - DALIRandomFlip:
          horizontal: true
          probability: 0.5
      - DALIRandomRotation:
          angle_range: [-10, 10]
          probability: 0.5
      - DALIBrightness:
          brightness_range: [0.8, 1.2]
          probability: 0.3
      # Continue with existing operators
      - EastRandomCropData:
          size: [640, 640]
          max_tries: 50
      - MakeBorderMap:
          shrink_ratio: 0.4
      # Replace NormalizeImage with DALINormalizeImage
      - DALINormalizeImage:
          scale: 1./255.
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
      - ToCHWImage:
      - KeepKeys:
          keep_keys: ['image', 'threshold_map', 'threshold_mask', 'shrink_map', 'shrink_mask']
```

### Example 2: Text Recognition with DALI

```yaml
Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/
    label_file_list:
      - ./train_data/train_list.txt
    transforms:
      # Replace DecodeImage with DALIDecodeImage
      - DALIDecodeImage:
          img_mode: BGR
          channel_first: False
          device: cpu
      - CTCLabelEncode:
      # Add DALI augmentation operators
      - DALIBrightness:
          brightness_range: [0.7, 1.3]
          probability: 0.4
      - DALIContrast:
          contrast_range: [0.7, 1.3]
          probability: 0.4
      - RecResizeImg:
          image_shape: [3, 32, 100]
      # Replace NormalizeImage with DALINormalizeImage
      - DALINormalizeImage:
          scale: 1./255.
          mean: [0.5, 0.5, 0.5]
          std: [0.5, 0.5, 0.5]
      - ToCHWImage:
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
```

## Configuration Files

Example configuration files are provided:

- **Detection**: `configs/det/det_r50_vd_db_dali.yml`
- **Recognition**: `configs/rec/rec_svtr_dali.yml`

## Performance Considerations

### When to Use DALI

DALI is most beneficial when:
- Training on GPU with sufficient GPU memory
- Data preprocessing is a bottleneck (CPU utilization is high)
- Using large batch sizes
- Applying multiple augmentation operations
- Working with high-resolution images

### Device Selection

- **CPU mode (`device: cpu`)**: Use when GPU memory is limited or for inference
- **GPU mode (`device: gpu`)**: Use for training to maximize throughput

### Best Practices

1. **Start with CPU mode**: Test your pipeline with `device: cpu` first
2. **Monitor GPU memory**: Switch to GPU mode if you have spare GPU memory
3. **Batch size**: Larger batch sizes benefit more from DALI acceleration
4. **Mixed pipeline**: You can mix DALI operators with standard PaddleOCR operators
5. **Validation**: Use DALI operators in both training and validation for consistency

## Compatibility

- DALI operators are designed to be compatible with existing PaddleOCR operators
- They follow the same interface and can be mixed with standard operators
- If DALI is not installed, operators will raise an ImportError with installation instructions
- All DALI operators are optional and backward compatible

## Limitations

Current implementation:
- DALI operators use the same data format as standard PaddleOCR operators
- Some operators still fall back to CPU/numpy for compatibility
- Full DALI pipeline (end-to-end GPU processing) is not yet implemented

## Troubleshooting

### DALI Not Found
```
ImportError: NVIDIA DALI is not installed
```
**Solution**: Install DALI using the installation instructions above

### CUDA Version Mismatch
```
RuntimeError: CUDA version mismatch
```
**Solution**: Install the DALI version matching your CUDA version

### Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: 
- Reduce batch size
- Use CPU mode (`device: cpu`)
- Reduce number of workers

## Future Enhancements

Planned improvements:
- Full DALI pipeline implementation for end-to-end GPU processing
- Additional DALI-specific operators
- Performance benchmarks and optimization guidelines
- Support for DALI external source for custom data loading

## References

- [NVIDIA DALI Documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)
- [DALI GitHub Repository](https://github.com/NVIDIA/DALI)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)

## Contributing

Contributions to improve DALI integration are welcome! Please follow the PaddleOCR contribution guidelines when submitting pull requests.
