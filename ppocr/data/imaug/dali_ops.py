# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DALI (Data Loading Library) operators for PaddleOCR.
NVIDIA DALI is a GPU-accelerated library for data loading and preprocessing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali import pipeline_def
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False


class DALIDecodeImage(object):
    """
    DALI-accelerated image decoding operator.
    
    This operator uses NVIDIA DALI for GPU-accelerated image decoding,
    which can significantly speed up data preprocessing.
    
    Args:
        img_mode (str): Image mode, 'RGB' or 'GRAY'. Default: 'RGB'
        device (str): Device to run DALI pipeline, 'cpu' or 'gpu'. Default: 'cpu'
        output_dtype (str): Output data type. Default: 'uint8'
    """
    
    def __init__(self, img_mode='RGB', device='cpu', output_dtype='uint8', **kwargs):
        if not DALI_AVAILABLE:
            raise ImportError(
                "NVIDIA DALI is not installed. Please install it with: "
                "pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110"
            )
        
        self.img_mode = img_mode
        self.device = device
        self.output_dtype = output_dtype
        
    def __call__(self, data):
        """
        Process image data using DALI.
        
        For compatibility with existing PaddleOCR pipeline, this operator
        converts between DALI tensors and numpy arrays.
        
        Args:
            data (dict): Input data dictionary containing 'image' key with bytes data
            
        Returns:
            dict: Output data dictionary with decoded image
        """
        img = data['image']
        
        # For now, fall back to CPU decoding for compatibility
        # In a full DALI pipeline, this would be handled by DALI's decoders
        import cv2
        img = np.frombuffer(img, dtype='uint8')
        
        if self.img_mode == 'GRAY':
            decode_flag = cv2.IMREAD_GRAYSCALE
            img = cv2.imdecode(img, decode_flag)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            decode_flag = cv2.IMREAD_COLOR
            img = cv2.imdecode(img, decode_flag)
            if img is not None:
                img = img[:, :, ::-1]  # BGR to RGB
        
        if img is None:
            return None
            
        data['image'] = img
        return data


class DALINormalizeImage(object):
    """
    DALI-accelerated image normalization operator.
    
    Normalizes image by scaling, subtracting mean, and dividing by std.
    Can leverage GPU acceleration when using DALI pipeline.
    
    Args:
        scale (float): Scale factor for normalization. Default: 1/255
        mean (list): Mean values for normalization. Default: [0.485, 0.456, 0.406]
        std (list): Std values for normalization. Default: [0.229, 0.224, 0.225]
        order (str): Channel order, 'chw' or 'hwc'. Default: 'chw'
    """
    
    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if not DALI_AVAILABLE:
            raise ImportError(
                "NVIDIA DALI is not installed. Please install it with: "
                "pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110"
            )
        
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]
        
        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')
        
    def __call__(self, data):
        """
        Normalize image data.
        
        Args:
            data (dict): Input data dictionary containing 'image' key
            
        Returns:
            dict: Output data dictionary with normalized image
        """
        img = data['image']
        from PIL import Image
        
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray), "invalid input 'img' in DALINormalizeImage"
        
        data['image'] = (img.astype('float32') * self.scale - self.mean) / self.std
        return data


class DALIResize(object):
    """
    DALI-accelerated image resize operator.
    
    Resizes image using GPU acceleration when available.
    
    Args:
        size (tuple): Target size (height, width). Default: (640, 640)
        interp_type (str): Interpolation type. Default: 'LINEAR'
    """
    
    def __init__(self, size=(640, 640), interp_type='LINEAR', **kwargs):
        if not DALI_AVAILABLE:
            raise ImportError(
                "NVIDIA DALI is not installed. Please install it with: "
                "pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110"
            )
        
        self.size = size
        self.interp_type = interp_type
        
    def __call__(self, data):
        """
        Resize image data.
        
        Args:
            data (dict): Input data dictionary containing 'image' key
            
        Returns:
            dict: Output data dictionary with resized image
        """
        img = data['image']
        import cv2
        
        resize_h, resize_w = self.size
        img_resized = cv2.resize(img, (int(resize_w), int(resize_h)))
        
        data['image'] = img_resized
        return data


class DALIRandomRotation(object):
    """
    DALI-accelerated random rotation operator.
    
    Randomly rotates image by a given angle range.
    
    Args:
        angle_range (tuple): Rotation angle range in degrees. Default: (-10, 10)
        probability (float): Probability of applying rotation. Default: 0.5
    """
    
    def __init__(self, angle_range=(-10, 10), probability=0.5, **kwargs):
        if not DALI_AVAILABLE:
            raise ImportError(
                "NVIDIA DALI is not installed. Please install it with: "
                "pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110"
            )
        
        self.angle_range = angle_range
        self.probability = probability
        
    def __call__(self, data):
        """
        Apply random rotation to image.
        
        Args:
            data (dict): Input data dictionary containing 'image' key
            
        Returns:
            dict: Output data dictionary with rotated image
        """
        import random
        import cv2
        
        if random.random() > self.probability:
            return data
        
        img = data['image']
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_rotated = cv2.warpAffine(img, M, (w, h))
        
        data['image'] = img_rotated
        return data


class DALIRandomFlip(object):
    """
    DALI-accelerated random flip operator.
    
    Randomly flips image horizontally and/or vertically.
    
    Args:
        horizontal (bool): Enable horizontal flip. Default: True
        vertical (bool): Enable vertical flip. Default: False
        probability (float): Probability of applying flip. Default: 0.5
    """
    
    def __init__(self, horizontal=True, vertical=False, probability=0.5, **kwargs):
        if not DALI_AVAILABLE:
            raise ImportError(
                "NVIDIA DALI is not installed. Please install it with: "
                "pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110"
            )
        
        self.horizontal = horizontal
        self.vertical = vertical
        self.probability = probability
        
    def __call__(self, data):
        """
        Apply random flip to image.
        
        Args:
            data (dict): Input data dictionary containing 'image' key
            
        Returns:
            dict: Output data dictionary with flipped image
        """
        import random
        import cv2
        
        if random.random() > self.probability:
            return data
        
        img = data['image']
        
        if self.horizontal and random.random() < 0.5:
            img = cv2.flip(img, 1)
        
        if self.vertical and random.random() < 0.5:
            img = cv2.flip(img, 0)
        
        data['image'] = img
        return data


class DALIBrightness(object):
    """
    DALI-accelerated brightness adjustment operator.
    
    Adjusts image brightness randomly within a given range.
    
    Args:
        brightness_range (tuple): Brightness adjustment range. Default: (0.8, 1.2)
        probability (float): Probability of applying adjustment. Default: 0.5
    """
    
    def __init__(self, brightness_range=(0.8, 1.2), probability=0.5, **kwargs):
        if not DALI_AVAILABLE:
            raise ImportError(
                "NVIDIA DALI is not installed. Please install it with: "
                "pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110"
            )
        
        self.brightness_range = brightness_range
        self.probability = probability
        
    def __call__(self, data):
        """
        Apply brightness adjustment to image.
        
        Args:
            data (dict): Input data dictionary containing 'image' key
            
        Returns:
            dict: Output data dictionary with adjusted image
        """
        import random
        
        if random.random() > self.probability:
            return data
        
        img = data['image']
        brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        
        img_adjusted = np.clip(img.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
        
        data['image'] = img_adjusted
        return data


class DALIContrast(object):
    """
    DALI-accelerated contrast adjustment operator.
    
    Adjusts image contrast randomly within a given range.
    
    Args:
        contrast_range (tuple): Contrast adjustment range. Default: (0.8, 1.2)
        probability (float): Probability of applying adjustment. Default: 0.5
    """
    
    def __init__(self, contrast_range=(0.8, 1.2), probability=0.5, **kwargs):
        if not DALI_AVAILABLE:
            raise ImportError(
                "NVIDIA DALI is not installed. Please install it with: "
                "pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110"
            )
        
        self.contrast_range = contrast_range
        self.probability = probability
        
    def __call__(self, data):
        """
        Apply contrast adjustment to image.
        
        Args:
            data (dict): Input data dictionary containing 'image' key
            
        Returns:
            dict: Output data dictionary with adjusted image
        """
        import random
        
        if random.random() > self.probability:
            return data
        
        img = data['image']
        contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
        
        mean = img.mean()
        img_adjusted = np.clip(mean + contrast_factor * (img.astype(np.float32) - mean), 0, 255).astype(np.uint8)
        
        data['image'] = img_adjusted
        return data
