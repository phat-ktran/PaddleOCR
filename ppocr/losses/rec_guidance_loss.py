# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from typing import Dict, List, Any
import paddle
from paddle import nn


class GuidanceLoss(nn.Layer):
    def __init__(self, mappings_path: str, **kwargs):
        super(GuidanceLoss, self).__init__()
        self.mappings = self._load_mappings(mappings_path)
        # Pre-compute max vocab size for efficiency
        self.max_vocab_idx = (
            max(
                max(char_indices)
                for char_indices in self.mappings.values()
                if char_indices
            )
            if self.mappings
            else 0
        )

    def _load_mappings(self, mappings_path: str) -> Dict[str, List[int]]:
        with open(mappings_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _create_target_distributions_batch(
        self, translations: List[str], masks: List[List[int]], vocab_size: int
    ) -> paddle.Tensor:
        all_target_dists = []

        for translation, mask in zip(translations, masks):
            if len(mask) == 0:
                continue

            # Create target distribution for this sample
            sample_target_dist = paddle.zeros([len(mask), vocab_size])

            for i, idx in enumerate(mask):
                if idx < len(translation):
                    mapped_chars = self.mappings.get(translation[idx], [])
                    if mapped_chars:
                        # Filter valid character indices
                        valid_chars = [
                            char_idx
                            for char_idx in mapped_chars
                            if char_idx < vocab_size
                        ]
                        if valid_chars:
                            uniform_prob = 1.0 / len(valid_chars)
                            sample_target_dist[i, valid_chars] = uniform_prob

            all_target_dists.append(sample_target_dist)

        if all_target_dists:
            return paddle.concat(all_target_dists, axis=0)
        else:
            return paddle.zeros([0, vocab_size])

    def _extract_predictions_batch(
        self, predicts: paddle.Tensor, masks: List[List[int]]
    ) -> paddle.Tensor:
        all_pred_subsets = []

        for batch_idx, mask in enumerate(masks):
            if len(mask) == 0:
                continue
            # Extract predictions for masked positions in this batch element
            pred_subset = predicts[mask, batch_idx, :]  # Shape: [len(mask), vocab_size]
            all_pred_subsets.append(pred_subset)

        if all_pred_subsets:
            return paddle.concat(all_pred_subsets, axis=0)
        else:
            return paddle.zeros([0, predicts.shape[2]])

    def _compute_kl_divergence(
        self, log_probs: paddle.Tensor, target_probs: paddle.Tensor
    ) -> paddle.Tensor:
        """Compute KL divergence between log probabilities and target probabilities."""
        eps = 1.0e-10
        loss = target_probs * (paddle.log(target_probs + eps) - log_probs)
        return loss

    def forward(
        self, predicts: Any, batch: List[paddle.Tensor]
    ) -> Dict[str, paddle.Tensor]:
        """
        Forward pass of guidance loss.

        Args:
            predicts: Model predictions
            batch: Batch data containing labels, lengths, masks, and translations

        Returns:
            Dictionary containing the computed loss
        """
        # Handle different predict formats
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]

        # Transpose to [N, B, vocab_size] format
        predicts = predicts.transpose((1, 0, 2))
        N, B, vocab_size = predicts.shape

        # Extract batch components
        batch[1].astype("int32")
        batch[2].astype("int64")
        label_masks = batch[3].astype("int64")
        label_translations = batch[4]

        # Convert masks to list format for easier processing
        masks_list = [
            mask.numpy().tolist() if hasattr(mask, "numpy") else mask
            for mask in label_masks
        ]

        # Count total valid samples and positions
        valid_samples = sum(1 for mask in masks_list if len(mask) > 0)
        total_positions = sum(len(mask) for mask in masks_list)

        if total_positions == 0:
            # No valid positions to compute loss
            return {"loss": paddle.zeros([1])}

        # Batch create target distributions
        target_dists = self._create_target_distributions_batch(
            label_translations, masks_list, vocab_size
        )

        # Batch extract predictions
        pred_subsets = self._extract_predictions_batch(predicts, masks_list)

        if target_dists.shape[0] == 0 or pred_subsets.shape[0] == 0:
            return {"loss": paddle.zeros([1])}

        # Apply log softmax to predictions
        pred_log_probs = paddle.nn.functional.log_softmax(pred_subsets, axis=-1)

        # Compute KL divergence loss in batch
        loss = self._compute_kl_divergence(pred_log_probs, target_dists)

        # Normalize by number of valid samples
        if valid_samples > 0:
            loss = loss / valid_samples

        return {"loss": loss}
