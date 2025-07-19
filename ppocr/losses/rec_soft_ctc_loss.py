import paddle
import numpy as np

from ppocr.data.imaug.utils.connections import BatchConnections


# Utility function to log tensor statistics
def log_tensor_stats(tensor, name, step=None):
    if tensor is None:
        print(f"[DEBUG] {name} is None")
        return
    if isinstance(tensor, list) or isinstance(tensor, np.ndarray):
        print(
            f"[DEBUG] {name} is a list or ndarray and cannot be converted to a tensor directly."
        )
        return
    stats = {
        "name": name,
        "shape": tensor.shape,
        "min": float(paddle.min(tensor)),
        "max": float(paddle.max(tensor)),
        "mean": float(paddle.mean(tensor)),
        "has_nan": bool(paddle.any(paddle.isnan(tensor))),
        "has_inf": bool(paddle.any(paddle.isinf(tensor))),
    }
    step_info = f" at step {step}" if step is not None else ""
    print(
        f"[DEBUG] {name}{step_info}: shape={stats['shape']}, min={stats['min']:.4e}, "
        f"max={stats['max']:.4e}, mean={stats['mean']:.4e}, "
        f"has_nan={stats['has_nan']}, has_inf={stats['has_inf']}"
    )


# Utility function to assert tensor validity
def assert_tensor_valid(tensor, name):
    if paddle.any(paddle.isnan(tensor)):
        print(f"[DEBUG] {name} contains NaN values: {tensor}")
        raise ValueError(f"{name} contains NaN values")
    if paddle.any(paddle.isinf(tensor)):
        print(f"[DEBUG] {name} contains Inf values: {tensor}")
        raise ValueError(f"{name} contains Inf values")


class SoftCTCLossFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, logits, connections, labels, norm_step=10, zero_infinity=False):
        full_probs = paddle.nn.functional.softmax(logits, axis=1)
        full_probs[full_probs == 0] = 1e-37

        N, C, T = full_probs.shape
        L = connections.size()

        alphas = paddle.zeros((N, L, T), dtype=logits.dtype)

        # Gather log-probabilities for CTC path
        probs = paddle.take_along_axis(
            full_probs, labels.unsqueeze(-1).expand([-1, -1, T]), axis=1
        )  # [N, L, T]

        alphas[:, :, 0] = connections.forward_start * probs[:, :, 0]

        c = paddle.sum(alphas[:, :, 0], axis=1)
        alphas[:, :, 0] /= c.reshape((N, 1))
        ll_forward = paddle.log(c)

        current_vector = alphas[:, :, 0]
        for t in range(1, T):
            current_vector = paddle.bmm(current_vector.reshape((N, 1, -1)), connections.forward).reshape((N, -1)) * probs[:, :, t]

            if t % norm_step == 0:
                c = paddle.sum(current_vector, axis=1)
                current_vector /= c.reshape((N, 1))
                ll_forward += paddle.log(c)

            alphas[:, :, t] = current_vector

        c = paddle.sum(alphas[:, :, -1] * connections.forward_end, axis=1)
        alphas[:, :, -1] /= c.reshape((N, 1))
        ll_forward += paddle.log(c)

        zero_mask = None
        if zero_infinity:
            zero_mask = paddle.logical_or(paddle.isnan(ll_forward), paddle.isinf(ll_forward))
            ll_forward[zero_mask] = 0
            
        ctx.save_for_backward(logits, labels)  # saves two tensors
        
        ctx.connections = connections
        ctx.full_probs = full_probs
        ctx.probs = probs
        ctx.alphas = alphas
        ctx.norm_step = norm_step
        ctx.zero_infinity = zero_infinity
        ctx.zero_mask = zero_mask
        return -ll_forward

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels = ctx.saved_tensor()  # note: singular in paddle
        connections = ctx.connections
        full_probs = ctx.full_probs
        probs = ctx.probs
        alphas = ctx.alphas
        norm_step = ctx.norm_step
        zero_infinity = ctx.zero_infinity
        zero_mask = ctx.zero_mask
    
        N, C, T = full_probs.shape
        L = connections.size()
    
        betas = paddle.zeros_like(alphas)
    
        # Initialization at final timestep
        betas[:, :, -1] = connections.backward_start * probs[:, :, -1]
        c = paddle.sum(betas[:, :, -1], axis=1)
        betas[:, :, -1] /= c.reshape([N, 1])
        ll_backward = paddle.log(c)
    
        current_vector = betas[:, :, -1]
        for t in range(T - 2, -1, -1):
            current_vector = (
                paddle.bmm(current_vector.reshape([N, 1, -1]), connections.backward)
                .reshape([N, -1]) * probs[:, :, t]
            )
    
            if t % norm_step == 0:
                c = paddle.sum(current_vector, axis=1)
                current_vector /= c.reshape([N, 1])
                ll_backward += paddle.log(c)
    
            betas[:, :, t] = current_vector
    
        c = paddle.sum(betas[:, :, 0] * connections.backward_end, axis=1)
        betas[:, :, 0] /= c.reshape([N, 1])
        ll_backward += paddle.log(c)
    
        # Gradient computation
        grad = paddle.zeros_like(logits)
        ab = alphas * betas  # [N, L, T]
    
        reshaped_labels = paddle.tile(paddle.reshape(labels, [N, L, 1]), repeat_times=[1, 1, T])  # [N, L, T]
    
        # Shape of grad: [N, C, T], reshaped_labels: [N, L, T], ab: [N, L, T]
        label_indices = paddle.reshape(reshaped_labels, [-1])  # [N * L * T]
        ab_updates = paddle.reshape(ab, [-1])  # [N * L * T]
    
        batch_idx = paddle.arange(N).reshape([N, 1, 1])
        batch_idx = paddle.tile(batch_idx, [1, L, T]).reshape([-1])  # [N * L * T]
    
        time_idx = paddle.arange(T).reshape([1, 1, T])
        time_idx = paddle.tile(time_idx, [N, L, 1]).reshape([-1])  # [N * L * T]
    
        indices = paddle.stack([batch_idx, label_indices, time_idx], axis=1)  # [N * L * T, 3]
        grad = paddle.scatter_nd_add(grad, index=indices, updates=ab_updates)
    
        ab = ab / probs
    
        ab_sum = paddle.sum(ab, axis=1, keepdim=True)  # [N, 1, T]
        denominator = full_probs * ab_sum  # [N, C, T]
        denominator = paddle.where(denominator == 0, paddle.full_like(denominator, 1e-37), denominator)
    
        grad = full_probs - grad / denominator
    
        # Handle zero_infinity
        if zero_infinity:
            if zero_mask is not None:
                grad = paddle.where(zero_mask.unsqueeze(1), paddle.zeros_like(grad), grad)
            # Avoid NaN/Inf (per-sample)
            is_nan_or_inf = paddle.logical_or(paddle.isnan(grad), paddle.isinf(grad))
            mask_per_sample = paddle.any(is_nan_or_inf, axis=[1, 2])
            for n in range(N):
                if mask_per_sample[n]:
                    grad[n] = 0
    
        grad = grad / N
    
        # Free memory
        del ctx.connections
        del ctx.full_probs
        del ctx.probs
        del ctx.alphas
        del ctx.norm_step
        del ctx.zero_infinity
        del ctx.zero_mask
    
        # Return grad for each input to forward()
        return grad


class SoftCTCLoss(paddle.nn.Layer):
    def __init__(self, norm_step=5, zero_infinity=False):
        super(SoftCTCLoss, self).__init__()
        self.norm_step = norm_step
        self.zero_infinity = zero_infinity

    def forward(self, logits, batch):
        labels, connections = batch[1], batch[3]
        batch_connections = BatchConnections.stack_connections(connections)
        for index, l in enumerate(labels):
            labels_padding = [0] * (batch_connections.size() - len(l))
            labels[index] = paddle.concat(
                [
                    labels[index],
                    paddle.to_tensor(labels_padding, dtype=labels[index].dtype),
                ]
            )
            labels[index] = paddle.to_tensor(labels[index], dtype=paddle.int64)
        labels = paddle.stack(labels)
        logits =  logits.transpose((0, 2, 1))  # B, V, T
        batch_connections = batch_connections.paddle()
        logits = paddle.where(paddle.isnan(logits), paddle.zeros_like(logits), logits)
        loss = SoftCTCLossFunction.apply(
            logits, batch_connections, labels, self.norm_step, self.zero_infinity
        )

        return {"loss": loss}