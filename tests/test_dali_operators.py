#!/usr/bin/env python3
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
Test script for DALI operators in PaddleOCR.

This script tests the DALI operator implementations without requiring
DALI to be installed (tests the fallback behavior).
"""

import sys
import os
import unittest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDALIOperators(unittest.TestCase):
    """Test DALI operators"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Import the dali_ops module directly to avoid paddle dependency
        import importlib.util
        dali_ops_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ppocr', 'data', 'imaug', 'dali_ops.py'
        )
        spec = importlib.util.spec_from_file_location("dali_ops", dali_ops_path)
        dali_ops = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dali_ops)
        self.dali_ops = dali_ops
        
        # Create a simple test image (100x100 RGB)
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Convert to bytes for DecodeImage
        import cv2
        _, encoded = cv2.imencode('.jpg', self.test_image)
        self.test_image_bytes = encoded.tobytes()
    
    def test_dali_availability_flag(self):
        """Test that DALI_AVAILABLE flag is set correctly"""
        self.assertIsInstance(self.dali_ops.DALI_AVAILABLE, bool)
        print(f"DALI_AVAILABLE: {self.dali_ops.DALI_AVAILABLE}")
    
    def test_operators_importable(self):
        """Test that all DALI operators can be imported"""
        operators = [
            'DALIDecodeImage',
            'DALINormalizeImage',
            'DALIResize',
            'DALIRandomRotation',
            'DALIRandomFlip',
            'DALIBrightness',
            'DALIContrast',
        ]
        
        for op_name in operators:
            self.assertTrue(
                hasattr(self.dali_ops, op_name),
                f"Operator {op_name} should be importable"
            )
    
    def test_operator_instantiation_without_dali(self):
        """Test that operators raise ImportError when DALI is not installed"""
        if self.dali_ops.DALI_AVAILABLE:
            self.skipTest("DALI is installed, skipping test for missing DALI")
        
        with self.assertRaises(ImportError) as cm:
            self.dali_ops.DALIDecodeImage()
        
        self.assertIn("NVIDIA DALI is not installed", str(cm.exception))
        self.assertIn("pip install", str(cm.exception))
    
    def test_decode_image_with_dali(self):
        """Test DALIDecodeImage operator if DALI is installed"""
        if not self.dali_ops.DALI_AVAILABLE:
            self.skipTest("DALI is not installed")
        
        op = self.dali_ops.DALIDecodeImage(img_mode='RGB')
        data = {'image': self.test_image_bytes}
        result = op(data)
        
        self.assertIsNotNone(result)
        self.assertIn('image', result)
        self.assertIsInstance(result['image'], np.ndarray)
        self.assertEqual(len(result['image'].shape), 3)
    
    def test_normalize_image_with_dali(self):
        """Test DALINormalizeImage operator if DALI is installed"""
        if not self.dali_ops.DALI_AVAILABLE:
            self.skipTest("DALI is not installed")
        
        op = self.dali_ops.DALINormalizeImage()
        data = {'image': self.test_image.copy()}
        result = op(data)
        
        self.assertIsNotNone(result)
        self.assertIn('image', result)
        self.assertIsInstance(result['image'], np.ndarray)
        # Normalized image should be float32
        self.assertEqual(result['image'].dtype, np.float32)
    
    def test_resize_with_dali(self):
        """Test DALIResize operator if DALI is installed"""
        if not self.dali_ops.DALI_AVAILABLE:
            self.skipTest("DALI is not installed")
        
        target_size = (50, 50)
        op = self.dali_ops.DALIResize(size=target_size)
        data = {'image': self.test_image.copy()}
        result = op(data)
        
        self.assertIsNotNone(result)
        self.assertIn('image', result)
        self.assertEqual(result['image'].shape[:2], target_size)
    
    def test_random_flip_with_dali(self):
        """Test DALIRandomFlip operator if DALI is installed"""
        if not self.dali_ops.DALI_AVAILABLE:
            self.skipTest("DALI is not installed")
        
        op = self.dali_ops.DALIRandomFlip(horizontal=True, probability=1.0)
        data = {'image': self.test_image.copy()}
        result = op(data)
        
        self.assertIsNotNone(result)
        self.assertIn('image', result)
        self.assertEqual(result['image'].shape, self.test_image.shape)
    
    def test_random_rotation_with_dali(self):
        """Test DALIRandomRotation operator if DALI is installed"""
        if not self.dali_ops.DALI_AVAILABLE:
            self.skipTest("DALI is not installed")
        
        op = self.dali_ops.DALIRandomRotation(angle_range=(-10, 10), probability=1.0)
        data = {'image': self.test_image.copy()}
        result = op(data)
        
        self.assertIsNotNone(result)
        self.assertIn('image', result)
        self.assertEqual(result['image'].shape, self.test_image.shape)
    
    def test_brightness_with_dali(self):
        """Test DALIBrightness operator if DALI is installed"""
        if not self.dali_ops.DALI_AVAILABLE:
            self.skipTest("DALI is not installed")
        
        op = self.dali_ops.DALIBrightness(brightness_range=(0.8, 1.2), probability=1.0)
        data = {'image': self.test_image.copy()}
        result = op(data)
        
        self.assertIsNotNone(result)
        self.assertIn('image', result)
        self.assertEqual(result['image'].shape, self.test_image.shape)
    
    def test_contrast_with_dali(self):
        """Test DALIContrast operator if DALI is installed"""
        if not self.dali_ops.DALI_AVAILABLE:
            self.skipTest("DALI is not installed")
        
        op = self.dali_ops.DALIContrast(contrast_range=(0.8, 1.2), probability=1.0)
        data = {'image': self.test_image.copy()}
        result = op(data)
        
        self.assertIsNotNone(result)
        self.assertIn('image', result)
        self.assertEqual(result['image'].shape, self.test_image.shape)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDALIOperators)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
