# copyright (c) 2025 PaddlePaddle Authors.
# Licensed under the Apache 2.0 licence.

from __future__ import absolute_import, division, print_function
import numpy as np
import paddle
from paddle import nn

__all__ = ["KCTCLoss"]


class KCTCLoss(nn.Layer):
    """
    CTC loss that supports *K* candidate transcripts **per image**.

    Rule:
      1. Every image i in the mini-batch can carry k_i candidate label
         sequences  (k_i ≥ 1).  The list is provided in batch[3]
         under the name ``k_ctc_labels`` (see dataset changes below).
      2. A single forward() duplicates the network prediction of
         image *i* k_i times so that we obtain one log-probability
         path for every candidate transcript.
      3. We concatenate all (prediction, candidate) pairs in the
         mini‐batch → call Paddle's warp-CTC **once**.
      4. The vector of losses (length ∑ k_i) is then split back into
         the original groups; we average inside every group,
         followed by an average over all images in the batch.
      5. Optional focal re-weighting (same flag as the stock CTCLoss).
    """

    def __init__(self, use_focal_loss: bool = False, **kwargs):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=0, reduction="none")
        self.use_focal_loss = use_focal_loss

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, predicts, batch):
        """
        Args
        ----
        predicts : Tensor  (B, T, C) or list/tuple last element
        batch    : tuple   (see dataloader)
                   ├─ batch[1]  = labels           (not used here)
                   ├─ batch[2]  = label_lengths    (not used here)
                   └─ batch[3]  = k_ctc_labels     (list of length B,
                            each element = List[np.ndarray] already
                            encoded w.r.t. the same dict as labels)
        """
        # ------------------------------------------------------------------
        # 0. prepare predictions  ------------------------------------------------
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]  # (B,T,C)

        B, T, C = predicts.shape
        k_ctc_labels = batch[3]  # (B, max_candidates, max_text_len)

        # ------------------------------------------------------------------
        # 1. collect candidates and build K vector  -------------------------
        K = []  # number of candidates per image: [k1, k2, ..., kB]
        label_flat = []  # concatenated labels
        label_len = []  # len of every label
        group_pos = []  # [ (start,end) ] indices after concat
        cursor = 0

        for b in range(B):
            cand_array = k_ctc_labels[b]  # (max_candidates, max_text_len)

            # Find actual number of candidates (non-zero candidates)
            # A candidate is considered valid if it has at least one non-zero token
            valid_candidates = []
            for i in range(cand_array.shape[0]):
                candidate = cand_array[i]
                # Convert to numpy for checking
                candidate_np = (
                    candidate
                    if isinstance(candidate, np.ndarray)
                    else candidate.numpy()
                )
                if np.any(candidate_np != 0):  # has non-zero tokens
                    valid_candidates.append(candidate_np)

            k_i = len(valid_candidates)
            if k_i == 0:
                raise ValueError("Sample {} carries zero candidate labels!".format(b))

            K.append(k_i)  # Store number of candidates for this image

            # collect labels - use pre-padded labels directly
            for candidate in valid_candidates:
                candidate_array = np.asarray(candidate, dtype="int32")

                # Calculate actual length (excluding trailing padding zeros)
                actual_len = len(candidate_array)
                for i in range(len(candidate_array) - 1, -1, -1):
                    if candidate_array[i] != 0:
                        actual_len = i + 1
                        break
                # Ensure we have at least length 1 (even if it's all blanks)
                actual_len = max(1, actual_len)

                # Use the already-padded candidate directly
                label_flat.append(candidate_array)
                label_len.append(actual_len)

            # remember slice boundaries for this image
            group_pos.append((cursor, cursor + k_i))
            cursor += k_i

        # ------------------------------------------------------------------
        # 2. expand predictions efficiently using repeat_interleave  --------
        # Single vectorized operation to duplicate predictions
        preds_expanded = paddle.repeat_interleave(predicts, paddle.to_tensor(K), axis=0)

        # Now transpose to (T, B_expanded, C) as required by nn.CTCLoss
        preds_expanded = preds_expanded.transpose((1, 0, 2))

        # ------------------------------------------------------------------
        # 3. build label tensors (no re-padding needed)  ---------------------
        # Labels are already padded to max_text_len, use them directly
        labels_tensor = paddle.to_tensor(np.array(label_flat), dtype="int32")
        input_lengths_tensor = paddle.to_tensor([T] * cursor, dtype="int64")
        label_lengths_tensor = paddle.to_tensor(label_len, dtype="int64")

        # ------------------------------------------------------------------
        # 4. one warp-CTC call + (optional) focal  ---------------------------
        loss_vec = self.ctc(
            preds_expanded, labels_tensor, input_lengths_tensor, label_lengths_tensor
        )
        # loss_vec shape = (Σk_i,)

        if self.use_focal_loss:
            weight = (1.0 - paddle.exp(-loss_vec)) ** 2
            loss_vec = loss_vec * weight

        # ------------------------------------------------------------------
        # 5. regroup & final average  ----------------------------------------
        # Calculate group sizes from group_pos
        group_sizes = [end - start for start, end in group_pos]

        # Split loss_vec into groups efficiently using paddle.split
        loss_groups = paddle.split(loss_vec, num_or_sections=group_sizes, axis=0)

        # Calculate mean for each group
        sample_losses = [paddle.mean(group) for group in loss_groups]

        # Final average
        loss = paddle.mean(paddle.stack(sample_losses))  # scalar
        return {"loss": loss}
