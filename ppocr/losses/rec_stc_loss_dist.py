# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
        log_prob = math.log(prob)
        for l in range(L + 1):
            p1 = 2 * l - 1
            p2 = 2 * l
            c1 = g.add_node(False, l == L)
            idx = star_idx if l == L else (star_idx + target[l])
            if p1 >= 0:
                g.add_arc(p1, c1, idx, idx, log_prob)
            g.add_arc(p2, c1, idx, idx, log_prob)
            g.add_arc(c1, c1, idx, idx, log_prob)
            if l < L:
                g.add_arc(c1, 2 * l + 1, target[l])
            g.add_arc(c1, p2, STC_BLANK_IDX)
        return g

    @staticmethod
    def forward(ctx, inputs, targets, prob, reduction="mean"):
        """Forward pass for Star CTC, dynamically choosing parallel or sequential execution."""
        B, T, Cstar = inputs.shape
        losses, scales, emissions_graphs = [None] * B, [None] * B, [None] * B
        C = Cstar // 2

        # This function encapsulates the processing for a single item in the batch.
        def process(b):
            g_emissions = gtn.linear_graph(T, Cstar, gtn.Device(gtn.CPU), not inputs.stop_gradient)
            # Move data for the current item to CPU for GTN processing
            cpu_data = inputs[b].cpu().numpy()
            g_emissions.set_weights(cpu_data.ctypes.data)

            g_criterion = STCLossFunction.create_stc_graph(targets[b], C, prob)
            g_criterion.arc_sort(False)

            g_loss = gtn.negate(gtn.forward_score(gtn.compose(g_criterion, g_emissions)))

            scale = 1.0
            if reduction == "mean":
                scale = 1.0 / T if T > 0 else scale
            elif reduction not in ("none", "sum"):
                raise ValueError(f"invalid value for reduction '{reduction}'")

            losses[b] = g_loss
            scales[b] = scale
            emissions_graphs[b] = g_emissions

        # ====================================================================
        # DYNAMIC EXECUTION STRATEGY
        # Check if we are in a distributed environment.
        # - If world_size <= 1: Not distributed. Use gtn.parallel_for for potential
        #   speedup by using multiple CPU cores on a single machine.
        # - If world_size > 1: Distributed. Use a standard for-loop to avoid
        #   nested parallelism, which can cause deadlocks.
        # ====================================================================
        world_size = paddle.distributed.get_world_size()
        if world_size <= 1:
            gtn.parallel_for(process, range(B))
        else:
            for b in range(B):
                process(b)
        
        ctx.losses = losses
        ctx.scales = scales
        ctx.emissions_graphs = emissions_graphs
        ctx.in_shape = inputs.shape
        ctx.dtype = inputs.dtype
        ctx.place = inputs.place
        ctx.reduction = reduction

        loss_values = [losses[b].item() * scales[b] for b in range(B)]
        loss_tensor = paddle.to_tensor(loss_values, place=inputs.place, dtype=inputs.dtype)
        
        if reduction == "mean":
            return paddle.mean(loss_tensor)
        elif reduction == "sum":
            return paddle.sum(loss_tensor)
        else: # "none"
            return loss_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for Star CTC, dynamically choosing parallel or sequential execution."""
        losses = ctx.losses
        scales = ctx.scales
        emissions_graphs = ctx.emissions_graphs
        in_shape = ctx.in_shape
        reduction = ctx.reduction
        B, T, C = in_shape
        
        dtype_mapping = {
            'float32': np.float32,
            'float64': np.float64,
            'FP32': np.float32,
            'FP64': np.float64,
            paddle.float32: np.float32,
            paddle.float64: np.float64,
        }
        
        # Get numpy dtype, fallback to float32 if not found
        np_dtype = dtype_mapping.get(ctx.dtype, np.float32)
        if hasattr(ctx.dtype, 'name'):
            np_dtype = dtype_mapping.get(ctx.dtype.name, np.float32)
        
        input_grad = paddle.to_tensor(
            np.zeros([B, T, C], dtype=np_dtype), 
            place=ctx.place, 
            dtype=ctx.dtype
        )

        # This function encapsulates the backward processing for a single item.
        def process(b):
            gtn.backward(losses[b], False)
            emissions = emissions_graphs[b]
            grad = emissions.grad().weights_to_numpy()
            item_grad = paddle.to_tensor(grad, place=ctx.place, dtype=ctx.dtype).reshape([T, C]) * scales[b]
            
            # Apply scaling from grad_output
            if reduction == "none":
                # For per-item loss, grad_output is a vector.
                input_grad[b] = item_grad * grad_output[b]
            else:
                # For 'mean' or 'sum', grad_output is a scalar.
                # The final scaling is applied outside the loop.
                input_grad[b] = item_grad

        # Use the same dynamic strategy as in the forward pass.
        world_size = paddle.distributed.get_world_size()
        if world_size <= 1:
            gtn.parallel_for(process, range(B))
        else:
            for b in range(B):
                process(b)

        # Apply final scaling based on reduction and grad_output.
        # In DDP, grad_output is 1.0 for the averaged loss.
        if reduction == 'mean':
            return input_grad * (grad_output / B)
        elif reduction == 'sum':
             return input_grad * grad_output
        else: # 'none'
            return input_grad

class STCLoss(nn.Layer):
    """
    Adaptable Star CTC Loss for PaddleOCR.
    
    This implementation automatically adapts its execution strategy:
    - In single-GPU mode, it uses `gtn.parallel_for` for CPU parallelization.
    - In multi-GPU (distributed) mode, it uses a standard loop to ensure stability.
    """
    
    def __init__(self, use_focal_loss=False, p0=1.0, plast=0.1, thalf=10000.0, **kwargs):
        super(STCLoss, self).__init__()
        self.use_focal_loss = use_focal_loss
        self.p0 = p0
        self.plast = plast
        self.thalf = thalf
        # Use a buffer to correctly save/load state and handle device placement.
        self.register_buffer('nstep', paddle.to_tensor(0, dtype='int64'))
        
        assert STC_BLANK_IDX == 0, "STC requires blank index to be 0"
    
    def _compute_probability(self):
        """Compute current probability based on training step."""
        if not self.training:
            return self.plast
        
        if self.nstep.item() == 0 and paddle.distributed.get_world_size() > 1:
            pass

        self.nstep += 1
        prob = self.plast + (self.p0 - self.plast) * math.exp(
            -self.nstep.item() * math.log(2) / self.thalf
        )
        return prob
    
    @staticmethod
    def logsubexp(a, b):
        """Compute log(exp(a) - exp(b)) in a numerically stable way."""
        B, T, C = b.shape
        a = a.tile([1, 1, C])
        return a + paddle.log1p(1e-7 - paddle.exp(b - a))
    
    def _prepare_inputs_for_stc(self, predicts, labels, label_lengths):
        """Prepares tensors for the STCLossFunction."""
        T, B, C = predicts.shape
        prob = self._compute_probability()
        lse = paddle.logsumexp(predicts[:, :, 1:], axis=2, keepdim=True)
        
        batch_targets, select_idx_set = [], set([STC_BLANK_IDX])
        for b in range(B):
            target_len = label_lengths[b].item()
            target_seq = labels[b, :target_len].tolist()
            batch_targets.append(target_seq)
            select_idx_set.update(target_seq)
        
        select_idx = sorted(list(select_idx_set))
        target_map = {t: i for i, t in enumerate(select_idx)}
        
        mapped_targets = [[target_map[t] for t in target] for target in batch_targets]
        
        select_idx_tensor = paddle.to_tensor(select_idx, place=predicts.place, dtype='int64')
        selected_predicts = paddle.index_select(predicts, select_idx_tensor, axis=2)
        
        if selected_predicts.shape[2] > 1:
            neglse = self.logsubexp(lse, selected_predicts[:, :, 1:])
        else:
            # Create an empty tensor for concatenation if only blank is present
            neglse = paddle.empty([T, B, 0], dtype=predicts.dtype, place=predicts.place)

        log_probs = paddle.concat([selected_predicts, lse, neglse], axis=2)
        return log_probs, mapped_targets, prob
    
    def forward(self, predicts, batch):
        """Forward pass following PaddleOCR's interface."""
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        
        # Input shape check and transpose to (T, B, C) if needed
        if predicts.shape[0] == batch[1].shape[0] and len(predicts.shape) == 3:
            predicts = predicts.transpose((1, 0, 2))
        
        log_probs = paddle.nn.functional.log_softmax(predicts, axis=-1)
        
        labels, label_lengths = batch[1].astype("int32"), batch[2].astype("int64")
        
        stc_inputs, stc_targets, prob = self._prepare_inputs_for_stc(
            log_probs, labels, label_lengths
        )
        
        stc_inputs = stc_inputs.transpose((1, 0, 2))
        
        if self.use_focal_loss:
            individual_losses = STCLossFunction.apply(stc_inputs, stc_targets, prob, "none")
            weight = paddle.exp(-individual_losses)
            weight = paddle.square(1.0 - weight)
            loss = paddle.mean(weight * individual_losses)
        else:
            loss = STCLossFunction.apply(stc_inputs, stc_targets, prob, "mean")
        
        return {"loss": loss}


# Test function to validate the implementation
def test_stc_loss():
    """Test the Star CTC loss with PaddleOCR-style inputs."""
    print(f"Running test in a non-distributed setting (world_size={paddle.distributed.get_world_size()}).")
    print("This will test the `gtn.parallel_for` execution path.")
    
    batch_size, time_steps, vocab_size, max_label_length = 4, 60, 501, 40
    
    predicts = paddle.randn([batch_size, time_steps, vocab_size])
    predicts.stop_gradient = False
    
    labels = paddle.randint(1, vocab_size, [batch_size, max_label_length])
    label_lengths = paddle.randint(1, max_label_length + 1, [batch_size], dtype="int64")

    batch = [None, labels, label_lengths]
    
    stc_loss = STCLoss(use_focal_loss=False)
    stc_loss.train()
    
    result = stc_loss(predicts, batch)
    loss = result['loss']
    
    print(f"\nStar CTC Loss: {loss.item():.4f}")
    print(f"Loss shape: {loss.shape}")
    
    loss.backward()
    print(f"Gradient calculated successfully. Grad shape: {predicts.grad.shape}")
    assert predicts.grad is not None
    
    # Test focal loss
    print("\n--- Testing with Focal Loss ---")
    stc_loss_focal = STCLoss(use_focal_loss=True)
    stc_loss_focal.train()
    result_focal = stc_loss_focal(predicts, batch)
    print(f"Star CTC Loss with Focal: {result_focal['loss'].item():.4f}")

if __name__ == "__main__":
    test_stc_loss()
    print("\nTest completed successfully.")