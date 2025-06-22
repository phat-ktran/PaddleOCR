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

import gtn
import paddle
import numpy as np
import math
from paddle import nn

STC_BLANK_IDX = 0

class STCLossFunction(paddle.autograd.PyLayer):
    @staticmethod
    def create_stc_graph(target, star_idx, prob):
        """Create Star CTC graph using GTN."""
        g = gtn.Graph(False)
        L = len(target)
        S = 2 * L + 1
        
        # Create nodes for standard CTC lattice
        for l in range(S):
            idx = (l - 1) // 2
            g.add_node(l == 0, l == S - 1 or l == S - 2)
            label = target[idx] if l % 2 else STC_BLANK_IDX
            if label == STC_BLANK_IDX:
                g.add_arc(l, l, label)
            if l > 0:
                g.add_arc(l - 1, l, label)
            if l % 2 and l > 1:
                g.add_arc(l - 2, l, label)

        # Add star nodes and connections
        for l in range(L + 1):
            p1 = 2 * l - 1
            p2 = 2 * l
            c1 = g.add_node(False, l == L)
            idx = star_idx if l == L else (star_idx + target[l])
            if p1 >= 0:
                g.add_arc(p1, c1, idx, idx, math.log(prob))
            g.add_arc(p2, c1, idx, idx, math.log(prob))
            g.add_arc(c1, c1, idx, idx, math.log(prob))
            if l < L:
                g.add_arc(c1, 2 * l + 1, target[l])
            g.add_arc(c1, p2, STC_BLANK_IDX)
        return g

    @staticmethod
    def forward(ctx, inputs, targets, prob, reduction="mean"):
        """Forward pass for Star CTC using GTN."""
        B, T, Cstar = inputs.shape
        losses, scales, emissions_graphs = [None] * B, [None] * B, [None] * B
        C = Cstar // 2

        def process(b):
            g_emissions = gtn.linear_graph(T, Cstar, gtn.Device(gtn.CPU), not inputs.stop_gradient)
            cpu_data = inputs[b].cpu().numpy()
            g_emissions.set_weights(cpu_data.ctypes.data)

            g_criterion = STCLossFunction.create_stc_graph(targets[b], C, prob)
            g_criterion.arc_sort(False)

            g_loss = gtn.negate(gtn.forward_score(gtn.compose(g_criterion, g_emissions)))

            scale = 1.0
            if reduction == "mean":
                scale = 1.0 / T if T > 0 else scale
            elif reduction != "none":
                raise ValueError(f"invalid value for reduction '{reduction}'")

            losses[b] = g_loss
            scales[b] = scale
            emissions_graphs[b] = g_emissions

        gtn.parallel_for(process, range(B))

        ctx.losses = losses
        ctx.scales = scales
        ctx.emissions_graphs = emissions_graphs
        ctx.in_shape = inputs.shape
        ctx.dtype = inputs.dtype
        ctx.place = inputs.place

        loss_values = [losses[b].item() * scales[b] for b in range(B)]
        loss_tensor = paddle.to_tensor(loss_values, place=inputs.place)
        
        if reduction == "mean":
            return paddle.mean(loss_tensor)
        elif reduction == "sum":
            return paddle.sum(loss_tensor)
        else:
            return loss_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for Star CTC."""
        losses = ctx.losses
        scales = ctx.scales
        emissions_graphs = ctx.emissions_graphs
        in_shape = ctx.in_shape
        B, T, C = in_shape
        input_grad = paddle.zeros([B, T, C], dtype=ctx.dtype)

        def process(b):
            gtn.backward(losses[b], False)
            emissions = emissions_graphs[b]
            grad = emissions.grad().weights_to_numpy()
            input_grad[b] = paddle.to_tensor(grad, place=ctx.place).reshape([T, C]) * scales[b]

        gtn.parallel_for(process, range(B))

        # Handle different grad_output shapes
        if grad_output.ndim == 0:  # scalar
            grad_scale = grad_output
        else:  # tensor
            grad_scale = grad_output.mean()
            
        input_grad *= grad_scale / B
        return input_grad

class STCLoss(nn.Layer):
    """
    Star CTC Loss implementation for PaddleOCR.
    
    This implementation maintains the original GTN-based Star CTC algorithm
    while adapting the interface to work with PaddleOCR's training pipeline.
    """
    
    def __init__(self, use_focal_loss=False, p0=1.0, plast=0.1, thalf=10000.0, **kwargs):
        super(STCLoss, self).__init__()
        self.use_focal_loss = use_focal_loss
        self.p0 = p0  # Initial probability
        self.plast = plast  # Final probability
        self.thalf = thalf  # Half-life for probability decay
        self.nstep = 0  # Training step counter
        
        # Ensure blank index is correct
        assert STC_BLANK_IDX == 0, "STC requires blank index to be 0"
    
    def _compute_probability(self):
        """Compute current probability based on training step."""
        if not self.training:
            return self.plast
        
        self.nstep += 1
        prob = self.plast + (self.p0 - self.plast) * math.exp(
            -self.nstep * math.log(2) / self.thalf
        )
        return prob
    
    @staticmethod
    def logsubexp(a, b):
        """Compute log(exp(a) - exp(b)) in a numerically stable way."""
        B, T, C = b.shape
        a = a.tile([1, 1, C])
        return a + paddle.log1p(1e-7 - paddle.exp(b - a))
    
    def _prepare_inputs_for_stc(self, predicts, labels, label_lengths):
        """
        Prepare inputs for Star CTC computation.
        
        Args:
            predicts: (T, B, C) log probabilities
            labels: (B, S) label sequences  
            label_lengths: (B,) label lengths
            
        Returns:
            prepared inputs for STCLossFunction
        """
        T, B, C = predicts.shape
        
        # Get current probability parameter
        prob = self._compute_probability()
        
        # Compute log-sum-exp for normalization
        lse = paddle.logsumexp(predicts[:, :, 1:], axis=2, keepdim=True)
        
        # Create target sequences for each batch item
        batch_targets = []
        select_idx_set = set([STC_BLANK_IDX])
        
        # Convert label_lengths to CPU and then to numpy for safe access
        label_lengths_cpu = label_lengths.cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        
        for b in range(B):
            # Safe access using numpy array
            target_len = int(label_lengths_cpu[b])
            target_seq = labels_cpu[b, :target_len].tolist()
            batch_targets.append(target_seq)
            select_idx_set.update(target_seq)
        
        # Create index mapping
        select_idx = sorted(list(select_idx_set))
        target_map = {t: i for i, t in enumerate(select_idx)}
        
        # Map targets to new indices
        mapped_targets = []
        for target in batch_targets:
            mapped_targets.append([target_map[t] for t in target])
        
        # Select relevant columns from predictions
        select_idx_tensor = paddle.to_tensor(select_idx, place=predicts.place)
        selected_predicts = paddle.index_select(predicts, select_idx_tensor, axis=2)
        
        # Compute negative log-sum-exp for star tokens
        if selected_predicts.shape[2] > 1:  # Check if we have non-blank tokens
            neglse = self.logsubexp(lse, selected_predicts[:, :, 1:])
        else:
            # Handle case where only blank token exists
            neglse = lse
        
        # Concatenate: [selected_tokens, lse, neglse]
        log_probs = paddle.concat([selected_predicts, lse, neglse], axis=2)
        
        return log_probs, mapped_targets, prob
    
    def forward(self, predicts, batch):
        """
        Forward pass following PaddleOCR's interface.
        
        Args:
            predicts: model predictions (can be list/tuple, we take the last one)
            batch: batch data containing labels and lengths
                   batch[1]: labels (B, S)
                   batch[2]: label lengths (B,)
        
        Returns:
            dict: {"loss": computed_loss}
        """
        # Handle multi-output predictions (take the last one)
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        
        # Transpose to (T, B, C) format and apply log_softmax
        predicts = predicts.transpose((1, 0, 2))
        log_probs = paddle.nn.functional.log_softmax(predicts, axis=-1)
        
        # Extract labels and lengths from batch
        labels = batch[1].astype("int32")
        label_lengths = batch[2].astype("int64")
        
        # Add validation to prevent memory access issues
        batch_size = labels.shape[0]
        max_label_len = labels.shape[1]
        
        # Validate label_lengths
        if paddle.any(label_lengths < 0) or paddle.any(label_lengths > max_label_len):
            raise ValueError(f"Invalid label_lengths: must be in range [0, {max_label_len}]")
        
        # Ensure tensors are on the same device
        if labels.place != predicts.place:
            labels = labels.to(predicts.place)
        if label_lengths.place != predicts.place:
            label_lengths = label_lengths.to(predicts.place)
        
        try:
            # Prepare inputs for Star CTC
            stc_inputs, stc_targets, prob = self._prepare_inputs_for_stc(
                log_probs, labels, label_lengths
            )
            
            # Transpose back to (B, T, C) for STCLossFunction
            stc_inputs = stc_inputs.transpose((1, 0, 2))
            
            # Compute Star CTC loss
            loss = STCLossFunction.apply(stc_inputs, stc_targets, prob, "mean")
            
            # Apply focal loss if enabled
            if self.use_focal_loss:
                # For focal loss, we need individual losses
                individual_losses = STCLossFunction.apply(stc_inputs, stc_targets, prob, "none")
                weight = paddle.exp(-individual_losses)
                weight = paddle.subtract(paddle.to_tensor([1.0]), weight)
                weight = paddle.square(weight)
                loss = paddle.mean(paddle.multiply(individual_losses, weight))
            
            return {"loss": loss}
            
        except Exception as e:
            # Provide more detailed error information
            print(f"Error in STCLoss forward pass:")
            print(f"  Batch size: {batch_size}")
            print(f"  Predicts shape: {predicts.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Label lengths shape: {label_lengths.shape}")
            print(f"  Labels device: {labels.place}")
            print(f"  Label lengths device: {label_lengths.place}")
            print(f"  Predicts device: {predicts.place}")
            print(f"  Error: {str(e)}")
            raise e


# Test function for the Star CTC loss
def test_stc_loss():
    """Test the Star CTC loss with PaddleOCR-style inputs."""
    
    # Create test data
    batch_size = 2  # Use smaller batch size for testing
    time_steps = 60
    vocab_size = 10001  # Typical OCR vocabulary size
    max_label_length = 40
    
    # Create predictions (B, T, C)
    predicts = paddle.randn([batch_size, time_steps, vocab_size])
    
    # Create labels and lengths (avoid using blank token 0 in labels)
    labels = paddle.randint(1, vocab_size, [batch_size, max_label_length])
    
    # Generate realistic label_lengths
    label_lengths = paddle.randint(1, max_label_length + 1, [batch_size], dtype="int64")
    
    # Ensure labels are valid for the given lengths
    for i in range(batch_size):
        length = label_lengths[i].item()
        # Fill positions beyond length with blank token
        labels[i, length:] = 0

    # Create batch in PaddleOCR format
    batch = [None, labels, label_lengths]
    
    # Initialize Star CTC loss
    stc_loss = STCLoss(use_focal_loss=False, p0=1.0, plast=0.1, thalf=1000.0)
    
    # Test forward pass
    stc_loss.train()  # Set to training mode
    result = stc_loss(predicts, batch)
    
    print(f"Star CTC Loss: {result['loss'].item():.4f}")
    print(f"Loss shape: {result['loss'].shape}")
    print(f"Current training step: {stc_loss.nstep}")
    # Call _compute_probability without incrementing nstep again
    current_prob = stc_loss.plast + (stc_loss.p0 - stc_loss.plast) * math.exp(
        -stc_loss.nstep * math.log(2) / stc_loss.thalf
    )
    print(f"Current probability: {current_prob:.4f}")
    
    # Test with focal loss
    stc_loss_focal = STCLoss(use_focal_loss=True, p0=1.0, plast=0.1, thalf=1000.0)
    stc_loss_focal.train()
    result_focal = stc_loss_focal(predicts, batch)
    
    print(f"\nStar CTC Loss with Focal: {result_focal['loss'].item():.4f}")
    
    # Test probability decay over multiple steps
    print("\nTesting probability decay:")
    prob_test_loss = STCLoss(use_focal_loss=False, p0=1.0, plast=0.1, thalf=10000.0)
    prob_test_loss.train()
    for step in range(0, 50001, 10000):
        prob_test_loss.nstep = step
        prob = prob_test_loss._compute_probability() # nstep is incremented here
        print(f"Step {prob_test_loss.nstep}: probability = {prob:.4f}")
        
    return result['loss']

if __name__ == "__main__":
    # Run test
    print("Running STCLoss test with corrected CUDA memory access...")
    test_loss = test_stc_loss()
    print("\nTest completed successfully.")