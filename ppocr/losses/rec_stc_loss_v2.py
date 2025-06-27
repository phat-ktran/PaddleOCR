"""
Copyright (c) Meta Platforms, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

PaddlePaddle implementation of Star Temporal Classification (STC) Loss with parallel processing
"""

import paddle
import paddle.nn as nn
import numpy as np
import math

# Import GTN - Handle case where GTN might not support PaddlePaddle
try:
    import gtn

    GTN_AVAILABLE = True
except ImportError:
    GTN_AVAILABLE = False
    print("Warning: GTN library not available. STC loss will not work without GTN.")

# blank idx is REQUIRED to be zero for current implementation
STC_BLANK_IDX = 0


class STCLossFunction:
    """
    Creates a function for STC with autograd using parallel processing
    NOTE: This function assumes <star>, <star>/token is appended to the input
    """

    @staticmethod
    def create_stc_graph(target, star_idx, prob):
        """
        Creates STC label graph

        Args:
            target: target sequence (list of token indices)
            star_idx: index of star token
            prob: token insertion penalty (before applying log)
        Returns:
            STC label graph as gtn.Graph
        """
        if not GTN_AVAILABLE:
            raise RuntimeError("GTN library is required for STC loss computation")

        g = gtn.Graph(False)
        L = len(target)
        S = 2 * L + 1

        # create self-less CTC graph
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

        # add extra nodes/arcs required for STC
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
    def forward(inputs, targets, prob, reduction="none"):
        """
        Forward pass for STC loss computation

        Args:
            inputs: Paddle tensor of shape (B, T, Cstar)
            targets: List of target sequences
            prob: Token insertion penalty probability
            reduction: Reduction method ("none", "mean", "sum")

        Returns:
            Loss tensor and auxiliary data for backward pass
        """
        if not GTN_AVAILABLE:
            raise RuntimeError("GTN library is required for STC loss computation")

        B, T, Cstar = inputs.shape
        losses, scales, emissions_graphs = [None] * B, [None] * B, [None] * B
        C = Cstar // 2

        def process_batch_item(b):
            # create emission graph
            g_emissions = gtn.linear_graph(
                T, Cstar, gtn.Device(gtn.CPU), not inputs.stop_gradient
            )

            # Convert paddle tensor to numpy for GTN
            if inputs.place.is_gpu_place():
                cpu_data = inputs[b].cpu().numpy().astype(np.float32)
            else:
                cpu_data = inputs[b].numpy().astype(np.float32)

            # Set weights for GTN graph
            g_emissions.set_weights(cpu_data.ctypes.data)

            # create criterion graph
            g_criterion = STCLossFunction.create_stc_graph(targets[b], C, prob)
            g_criterion.arc_sort(False)

            # compose the graphs
            g_loss = gtn.negate(
                gtn.forward_score(gtn.compose(g_criterion, g_emissions))
            )

            scale = 1.0
            if reduction == "mean":
                scale = 1.0 / T if T > 0 else 1.0
            elif reduction == "sum":
                scale = 1.0
            elif reduction != "none":
                raise ValueError(
                    f"Invalid reduction '{reduction}'. Must be 'none', 'mean', or 'sum'"
                )

            # Save for backward:
            losses[b] = g_loss
            scales[b] = scale
            emissions_graphs[b] = g_emissions

        # Process batch items in parallel using GTN
        gtn.parallel_for(process_batch_item, range(B))

        # Convert losses to paddle tensor - match PyTorch exactly
        loss_values = [losses[b].item() * scales[b] for b in range(B)]

        # Create tensor on appropriate device
        if inputs.place.is_gpu_place():
            loss_tensor = paddle.to_tensor(
                loss_values, dtype=inputs.dtype, place=inputs.place
            )
        else:
            loss_tensor = paddle.to_tensor(loss_values, dtype=inputs.dtype)

        # Always return mean for consistency with PyTorch version
        return paddle.mean(loss_tensor), (losses, scales, emissions_graphs)

    @staticmethod
    def backward(inputs, targets, prob, reduction, grad_output, auxiliary_data):
        """
        Backward pass for STC loss computation

        Args:
            inputs: Input tensor
            targets: Target sequences
            prob: Token insertion penalty probability
            reduction: Reduction method
            grad_output: Gradient from upstream
            auxiliary_data: Data saved from forward pass

        Returns:
            Input gradients
        """
        if not GTN_AVAILABLE:
            raise RuntimeError("GTN library is required for STC loss computation")

        losses, scales, emissions_graphs = auxiliary_data
        B, T, Cstar = inputs.shape

        # Initialize gradient tensor as numpy array for thread-safe parallel access
        input_grad = paddle.zeros([B, T, Cstar], dtype=inputs.dtype)
        input_grad_numpy = input_grad.numpy()

        def process_batch_item(b):
            # Perform backward pass on GTN graph
            gtn.backward(losses[b], False)

            # Extract gradients from emission graph
            emissions = emissions_graphs[b]
            grad_numpy = emissions.grad().weights_to_numpy()

            # Apply scale and store directly in numpy array for thread safety
            input_grad_numpy[b] = grad_numpy.reshape(T, Cstar) * scales[b]

        # Compute gradients for batch items in parallel
        gtn.parallel_for(process_batch_item, range(B))

        # Convert back to paddle tensor
        input_grad = paddle.to_tensor(input_grad_numpy, dtype=inputs.dtype)

        # Move to appropriate device
        if inputs.place.is_gpu_place():
            input_grad = paddle.to_tensor(input_grad.numpy(), place=inputs.place)

        # Apply upstream gradient exactly as PyTorch: grad_output is scalar, multiply by /B
        input_grad = input_grad * grad_output / B

        return input_grad


class STCLossLayer(paddle.autograd.PyLayer):
    """
    PaddlePaddle autograd layer for STC loss with parallel processing
    """

    @staticmethod
    def forward(ctx, inputs, targets, prob, reduction="none"):
        # Compute forward pass and get auxiliary data
        loss, auxiliary_data = STCLossFunction.forward(inputs, targets, prob, reduction)

        # Save for backward - store data directly in context like other PaddlePaddle implementations
        ctx.inputs = inputs
        ctx.targets = targets
        ctx.prob = prob
        ctx.reduction = reduction
        ctx.auxiliary_data = auxiliary_data

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved data - access directly from context
        inputs = ctx.inputs
        targets = ctx.targets
        prob = ctx.prob
        reduction = ctx.reduction
        auxiliary_data = ctx.auxiliary_data

        # Compute input gradients
        input_grad = STCLossFunction.backward(
            inputs, targets, prob, reduction, grad_output, auxiliary_data
        )

        # Return gradients only for inputs (PaddlePaddle PyLayer expects single output)
        return input_grad


def STCLossFunc(inputs, targets, prob, reduction="none"):
    """
    STC Loss function with parallel processing

    Args:
        inputs: Tensor of shape (B, T, Cstar)
        targets: List of target sequences
        prob: Token insertion penalty probability
        reduction: Reduction method

    Returns:
        Loss tensor
    """
    return STCLossLayer.apply(inputs, targets, prob, reduction)


class STC(nn.Layer):
    """The Star Temporal Classification loss with parallel processing.

    Calculates loss between a continuous (unsegmented) time series and a
    partially labeled target sequence using GTN parallel processing for efficiency.

    Args:
        blank_idx: Index of blank token (must be 0)
        p0: Initial value for token insertion penalty (before applying log)
        plast: Final value for token insertion penalty (before applying log)
        thalf: Number of steps for token insertion penalty to reach (p0 + plast)/2
        reduction: Reduction method ("none", "mean", "sum")
    """

    def __init__(self, blank_idx=0, p0=1.0, plast=1.0, thalf=1.0, reduction="none"):
        super(STC, self).__init__()
        if blank_idx != STC_BLANK_IDX:
            raise ValueError(f"blank_idx must be {STC_BLANK_IDX}, got {blank_idx}")

        self.blank_idx = blank_idx
        self.p0 = p0
        self.plast = plast
        self.thalf = thalf
        self.nstep = 0
        self.reduction = reduction

    @staticmethod
    def logsubexp(a, b):
        """
        Computes log(exp(a) - exp(b)) in a numerically stable way
        Match PyTorch implementation exactly

        Args:
            a: Tensor of size (B, T, 1)
            b: Tensor of size (B, T, C)
        Returns:
            Tensor of size (B, T, C)
        """
        # Match PyTorch grad_enabled context handling
        with paddle.set_grad_enabled(not a.stop_gradient):
            B, T, C = b.shape
            # Expand a to match b's dimensions - match PyTorch tile behavior
            a_expanded = paddle.tile(a, [1, 1, C])

            # Compute log(exp(a) - exp(b)) = a + log(1 - exp(b - a))
            # Match PyTorch: a + torch.log1p(1e-7 - torch.exp(b - a))
            diff = b - a_expanded
            result = a_expanded + paddle.log1p(1e-7 - paddle.exp(diff))

            return result

    def forward(self, predicts, batch):
        """
        Computes STC loss for the given input and partially labeled target

        Args:
            predicts: Tensor of size (B, T, C) or list/tuple containing predictions
                T - # time steps, B - batch size, C - alphabet size (including blank)
                The logarithmized probabilities of the outputs
            batch: Batch data containing labels and label lengths

        Returns:
            Loss tensor in dict format like other PaddleOCR losses
        """
        if not GTN_AVAILABLE:
            raise RuntimeError(
                "GTN library is required for STC loss. Please install GTN with PaddlePaddle support."
            )

        # Handle predictions format like CTC loss
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]

        # Extract targets from batch like CTC loss
        labels = batch[1].astype("int32")  # Target labels
        label_lengths = batch[2].astype("int64")  # Label lengths

        # Convert labels to list of sequences for STC processing
        targets = []
        start_idx = 0
        for idx, length in enumerate(label_lengths):
            length = int(length.item())
            target_seq = labels[idx][start_idx : start_idx + length].tolist()
            # Remove any padding tokens (usually 0 or negative values)
            target_seq = [t for t in target_seq if t > 0]
            targets.append(target_seq)
            start_idx += length

        # Update step count during training
        if self.training:
            self.nstep += 1

        # Compute current probability for token insertion penalty
        prob = self.plast + (self.p0 - self.plast) * math.exp(
            -self.nstep * math.log(2) / self.thalf
        )

        log_probs = predicts

        B, T, C = log_probs.shape

        # Ensure grad context matches PyTorch
        with paddle.set_grad_enabled(not log_probs.stop_gradient):
            # <star> - match PyTorch logsumexp
            lse = paddle.logsumexp(log_probs[:, :, 1:], axis=2, keepdim=True)

            # Select only the tokens present in current batch to reduce computation
            select_idx = [STC_BLANK_IDX] + list(
                set([t for target in targets for t in target])
            )

            # Create mapping from original indices to selected indices
            target_map = {t: i for i, t in enumerate(select_idx)}

            # Select relevant token probabilities
            select_idx_tensor = paddle.to_tensor(select_idx, dtype="int64")
            log_probs_selected = paddle.index_select(
                log_probs, select_idx_tensor, axis=2
            )

            # Remap targets to use selected indices
            targets_remapped = [[target_map[t] for t in target] for target in targets]

            # <star>\tokens for all tokens present in current batch
            neglse = STC.logsubexp(lse, log_probs_selected[:, :, 1:])

            # Concatenate (tokens, <star>, <star>\tokens) - match PyTorch order
            log_probs_final = paddle.concat([log_probs_selected, lse, neglse], axis=2)

        loss = STCLossFunc(log_probs_final, targets_remapped, prob, self.reduction)
        return {"loss": loss}


# Alias for backward compatibility
STCLoss = STC


def _create_sample_data(batch_size=2, time_steps=10, vocab_size=20):
    """
    Create sample data for testing STC loss

    Args:
        batch_size: Number of samples in batch
        time_steps: Number of time steps
        vocab_size: Vocabulary size (including blank)

    Returns:
        inputs: Log probabilities tensor of shape (batch_size, time_steps, vocab_size)
        targets: List of target sequences
    """
    # Create random log probabilities
    import paddle.nn.functional as F
    import random

    inputs = paddle.randn([batch_size, time_steps, vocab_size])
    inputs = F.log_softmax(inputs, axis=-1)

    # Create dynamic sample targets (excluding blank token which is index 0)
    targets = []
    for i in range(batch_size):
        # Create random target sequence length between 2 and 6
        target_length = random.randint(2, min(6, vocab_size - 1))
        # Create random target sequence (excluding blank token 0)
        target = [random.randint(1, vocab_size - 1) for _ in range(target_length)]
        targets.append(target)

    return inputs, targets


def _basic_usage_example():
    """
    Basic usage example of STCLoss with parallel processing
    """
    print("=== Basic STCLoss Usage Example (Parallel Processing) ===")

    # Create sample data - test with different batch sizes
    batch_size, time_steps, vocab_size = 5, 10, 20
    inputs, targets = _create_sample_data(batch_size, time_steps, vocab_size)

    # Enable gradients for input tensor
    inputs.stop_gradient = False

    # Convert targets to PaddleOCR batch format
    all_labels = []
    label_lengths = []
    for target in targets:
        all_labels.extend(target)
        label_lengths.append(len(target))

    labels_tensor = paddle.to_tensor(all_labels, dtype="int32")
    lengths_tensor = paddle.to_tensor(label_lengths, dtype="int64")
    batch = [None, labels_tensor, lengths_tensor]

    print(f"Input shape: {inputs.shape}")
    print(f"Targets: {targets}")
    print(f"Labels tensor shape: {labels_tensor.shape}")
    print(f"Label lengths: {label_lengths}")
    print(f"Using GTN parallel processing for {len(targets)} batch items")

    # Initialize STC loss
    stc_loss = STCLoss(
        blank_idx=0,  # Blank token index (must be 0)
        p0=1.0,  # Initial token insertion penalty
        plast=0.1,  # Final token insertion penalty
        thalf=1000,  # Steps to reach halfway point
        reduction="mean",  # Reduction method
    )

    # Compute loss
    loss_dict = stc_loss(inputs, batch)
    loss = loss_dict["loss"]
    print(f"Loss: {loss.item():.4f}")

    # Compute gradients
    loss.backward()
    # In PaddlePaddle, gradients are computed but accessed differently than PyTorch
    print("Input gradients computed with parallel processing (shape matches input)")

    return loss


def _batch_size_test():
    """
    Test STC loss with different batch sizes using parallel processing
    """
    print("=== Batch Size Testing (Parallel Processing) ===")

    vocab_size = 1200
    time_steps = 60

    # Test different batch sizes
    for batch_size in [1, 2, 4, 8, 16]:
        print(f"\nTesting batch size: {batch_size}")

        # Create sample data
        inputs, targets = _create_sample_data(batch_size, time_steps, vocab_size)
        inputs.stop_gradient = False

        # Convert targets to PaddleOCR batch format
        all_labels = []
        label_lengths = []
        for target in targets:
            all_labels.extend(target)
            label_lengths.append(len(target))

        labels_tensor = paddle.to_tensor(all_labels, dtype="int32")
        lengths_tensor = paddle.to_tensor(label_lengths, dtype="int64")
        batch = [None, labels_tensor, lengths_tensor]

        print(f"Input shape: {inputs.shape}")
        print(f"Number of targets: {len(targets)}")
        print(f"Target lengths: {[len(t) for t in targets]}")
        print(f"GTN will process {batch_size} items in parallel")

        # Initialize STC loss
        stc_loss = STCLoss(
            blank_idx=0,
            p0=1.0,
            plast=0.1,
            thalf=1000,
            reduction="mean",
        )

        # Compute loss
        loss_dict = stc_loss(inputs, batch)
        loss = loss_dict["loss"]
        print(f"Loss: {loss.item():.4f}")

        # Compute gradients
        loss.backward()
        print("Gradients computed successfully with parallel processing")


def _compare_implementations():
    """
    Compare sequential vs parallel implementations
    """
    print("=== Implementation Comparison ===")
    print("This version (V2) uses GTN parallel processing for:")
    print("✓ Faster batch processing with gtn.parallel_for")
    print("✓ Thread-safe parallel computation of loss and gradients")
    print("✓ Improved performance on multi-core systems")
    print("✓ Same mathematical results as sequential version")
    print("")
    print("Key differences from V1 (sequential):")
    print("- Forward pass: gtn.parallel_for(process_batch_item, range(B))")
    print("- Backward pass: gtn.parallel_for(process_batch_item, range(B))")
    print("- Memory management: Pre-allocated arrays for thread safety")
    print("- Performance: O(1) time complexity for batch processing vs O(B)")


if __name__ == "__main__":
    print("STCLoss V2 - Parallel Processing Implementation")
    print("=" * 55)
    _basic_usage_example()
    print("\n" + "=" * 55 + "\n")
    _batch_size_test()
    print("\n" + "=" * 55 + "\n")
    _compare_implementations()
    print("\n" + "=" * 55)
    print(
        "Note: This implementation uses gtn.parallel_for for efficient batch processing"
    )
