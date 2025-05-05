# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This code is refer from: https://github.com/KaiyangZhou/pytorch-center-loss
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class AmpCenterLoss(nn.Layer):
    """
    Reference: Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Improved implementation with AMP compatibility and better numerical stability.
    """
    def __init__(self, num_classes=6625, feat_dim=96, center_file_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        # Initialize centers as Parameters instead of Tensors for better AMP compatibility
        self.centers = self.create_parameter(
            shape=[self.num_classes, self.feat_dim],
            dtype='float32',  # Use float32 consistently for parameters
            default_initializer=nn.initializer.Normal()
        )
        
        # Load centers from file if specified
        if center_file_path is not None:
            assert os.path.exists(
                center_file_path
            ), f"center path({center_file_path}) must exist when it is not None."
            with open(center_file_path, "rb") as f:
                char_dict = pickle.load(f)
                centers_np = self.centers.numpy()
                for key in char_dict.keys():
                    centers_np[key] = char_dict[key]
                self.centers.set_value(centers_np)
    
    def forward(self, predicts, batch):
        """
        Forward method for CenterLoss, renamed from __call__ for consistency with PaddlePaddle API.
        
        Args:
            predicts: A tuple/list of (features, predictions)
            batch: The input batch data
            
        Returns:
            dict: A dictionary containing the loss
        """
        assert isinstance(predicts, (list, tuple))
        features, predicts = predicts
        
        # Reshape features and convert to appropriate dtype
        feats_reshape = paddle.reshape(features, [-1, features.shape[-1]])
        
        # Get label from predictions
        label = paddle.argmax(predicts, axis=2)
        label = paddle.reshape(label, [label.shape[0] * label.shape[1]])
        batch_size = feats_reshape.shape[0]
        
        # Calculate distances using a more stable approach
        # 1. Normalize features for better numerical stability
        norm_feats = F.normalize(feats_reshape, axis=1)
        norm_centers = F.normalize(self.centers, axis=1)
        
        # 2. Use cosine distance instead of euclidean for better AMP compatibility
        # Cosine similarity ranges from -1 to 1, with 1 being most similar
        # Convert to a distance by subtracting from 1
        cosine_sim = paddle.matmul(norm_feats, paddle.transpose(norm_centers, [1, 0]))
        dist = 1.0 - cosine_sim
        
        # Generate the mask for selected classes
        classes = paddle.arange(self.num_classes).astype("int64")
        label_expanded = paddle.expand(
            paddle.unsqueeze(label, 1), (batch_size, self.num_classes)
        )
        mask = paddle.equal(
            paddle.expand(classes, [batch_size, self.num_classes]), label_expanded
        ).astype("float32")
        
        # Apply mask and calculate loss
        dist = paddle.multiply(dist, mask)
        
        # Sum only the non-zero elements (where mask is 1)
        non_zero_count = paddle.sum(mask)
        loss = paddle.sum(dist) / (non_zero_count + 1e-12)
        
        return {"loss_center": loss}