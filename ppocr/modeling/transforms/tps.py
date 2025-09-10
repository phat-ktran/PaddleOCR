# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
"""
This code is adapted from:
https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/modules/transformation.py

Fixes / defensive changes:
- Validate image height/width to avoid zero-sized tensors.
- Use linspace for grid generation (robust for even/odd sizes).
- Upsample tiny images automatically (configurable minimum).
- Add informative error messages where tensors can become empty.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn, ParamAttr
from paddle.nn import functional as F
import numpy as np


class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        act=None,
        name=None,
    ):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
        )
        bn_name = "bn_" + name
        self.bn = nn.BatchNorm(
            out_channels,
            act=act,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(bn_name + "_offset"),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + "_variance",
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class LocalizationNetwork(nn.Layer):
    def __init__(self, in_channels, num_fiducial, loc_lr, model_name):
        super(LocalizationNetwork, self).__init__()
        self.F = num_fiducial
        F = num_fiducial
        if model_name == "large":
            num_filters_list = [64, 128, 256, 512]
            fc_dim = 256
        else:
            num_filters_list = [16, 32, 64, 128]
            fc_dim = 64

        self.block_list = []
        for fno in range(0, len(num_filters_list)):
            num_filters = num_filters_list[fno]
            name = "loc_conv%d" % fno
            conv = self.add_sublayer(
                name,
                ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=3,
                    act="relu",
                    name=name,
                ),
            )
            self.block_list.append(conv)
            if fno == len(num_filters_list) - 1:
                pool = nn.AdaptiveAvgPool2D(1)
            else:
                pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
            in_channels = num_filters
            self.block_list.append(pool)

        name = "loc_fc1"
        stdv = 1.0 / math.sqrt(num_filters_list[-1] * 1.0)
        self.fc1 = nn.Linear(
            in_channels,
            fc_dim,
            weight_attr=ParamAttr(
                learning_rate=loc_lr,
                name=name + "_w",
                initializer=nn.initializer.Uniform(-stdv, stdv),
            ),
            bias_attr=ParamAttr(name=name + ".b_0"),
            name=name,
        )

        # Init fc2
        initial_bias = self.get_initial_fiducials()
        initial_bias = initial_bias.reshape(-1)
        name = "loc_fc2"
        param_attr = ParamAttr(
            learning_rate=loc_lr,
            initializer=nn.initializer.Assign(np.zeros([fc_dim, F * 2])),
            name=name + "_w",
        )
        bias_attr = ParamAttr(
            learning_rate=loc_lr,
            initializer=nn.initializer.Assign(initial_bias),
            name=name + "_b",
        )
        self.fc2 = nn.Linear(
            fc_dim, F * 2, weight_attr=param_attr, bias_attr=bias_attr, name=name
        )
        self.out_channels = F * 2

    def forward(self, x):
        B = x.shape[0]
        for block in self.block_list:
            x = block(x)
    
        # safer flatten
        x = paddle.flatten(x, start_axis=1)  # [B, C]
    
        if paddle.in_dynamic_mode():
            if x.shape[1] == 0:
                raise ValueError(f"LocalizationNetwork fc input has 0 features! shape={x.shape}")
    
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.reshape(shape=[-1, self.F, 2])
        return x


    def get_initial_fiducials(self):
        """see RARE paper Fig. 6 (a)"""
        F = self.F
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return initial_bias



class GridGenerator(nn.Layer):
    def __init__(self, in_channels, num_fiducial):
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.F = num_fiducial

        name = "ex_fc"
        initializer = nn.initializer.Constant(value=0.0)
        param_attr = ParamAttr(
            learning_rate=0.0, initializer=initializer, name=name + "_w"
        )
        bias_attr = ParamAttr(
            learning_rate=0.0, initializer=initializer, name=name + "_b"
        )
        self.fc = nn.Linear(
            in_channels, 6, weight_attr=param_attr, bias_attr=bias_attr, name=name
        )

    def forward(self, batch_C_prime, I_r_size):
        """
        Generate the grid for the grid_sampler.
        Args:
            batch_C_prime: [B, F, 2]
            I_r_size: (H, W) or tensor/tuple of ints
        Return:
            batch_P_prime: [B, H*W, 2] grid for grid_sampler in float32
        """
        # Validate I_r_size
        if (
            I_r_size is None
            or len(I_r_size) != 2
            or I_r_size[0] is None
            or I_r_size[1] is None
        ):
            raise ValueError(f"I_r_size is invalid: {I_r_size}")

        I_r_height, I_r_width = int(I_r_size[0]), int(I_r_size[1])
        if I_r_height <= 0 or I_r_width <= 0:
            raise ValueError(f"I_r_size contains non-positive dimension: {I_r_size}")

        C = self.build_C_paddle()
        P = self.build_P_paddle((I_r_height, I_r_width))

        # Build tensors needed for transformation. Keep as float32 for grid_sample.
        inv_delta_C_tensor = self.build_inv_delta_C_paddle(C).astype("float32")
        P_hat_tensor = self.build_P_hat_paddle(C, paddle.to_tensor(P).astype("float32")).astype("float32")

        # Freeze these intermediate tensors
        inv_delta_C_tensor.stop_gradient = True
        P_hat_tensor.stop_gradient = True

        batch_C_ex_part_tensor = self.get_expand_tensor(batch_C_prime)
        batch_C_ex_part_tensor.stop_gradient = True

        # concat to get B x (F+3) x 2
        batch_C_prime_with_zeros = paddle.concat(
            [batch_C_prime, batch_C_ex_part_tensor], axis=1
        )  # B x F+3 x 2

        # Now compute T and P':
        # inv_delta_C_tensor: F+3 x F+3
        # We need to tile inv_delta_C to match batch size or use matmul broadcasting
        # Use paddle.matmul with appropriate broadcasting: (F+3, F+3) x (B, F+3, 2) -> (B, F+3, 2)
        batch_T = paddle.matmul(inv_delta_C_tensor, batch_C_prime_with_zeros)
        batch_P_prime = paddle.matmul(P_hat_tensor, batch_T)
        # Return as float32
        return batch_P_prime.astype("float32")

    def build_C_paddle(self):
        """Return coordinates of fiducial points in I_r; C as float32 (F x 2)"""
        F = self.F
        # Use linspace to avoid possible arange empty issues
        ctrl_pts_x = paddle.linspace(-1.0, 1.0, int(F / 2), dtype="float32")
        ctrl_pts_y_top = -1 * paddle.ones([int(F / 2)], dtype="float32")
        ctrl_pts_y_bottom = paddle.ones([int(F / 2)], dtype="float32")
        ctrl_pts_top = paddle.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = paddle.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = paddle.concat([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def build_P_paddle(self, I_r_size):
        """
        Build grid points P (n x 2) in double precision as used by original implementation.
        I_r_size: (height, width)
        """
        I_r_height, I_r_width = int(I_r_size[0]), int(I_r_size[1])

        if I_r_height <= 0 or I_r_width <= 0:
            raise ValueError(f"Invalid target grid size: {I_r_size}")

        # Build normalized grid in [-1, 1] using linspace which is robust and produces exactly I_r_width / I_r_height points
        I_r_grid_x = paddle.linspace(-1.0, 1.0, I_r_width, dtype="float32")
        I_r_grid_y = paddle.linspace(-1.0, 1.0, I_r_height, dtype="float32")

        # meshgrid: produces H x W x 2, we then reshape to (n, 2)
        # Use paddle.meshgrid which returns sequence [X, Y] where X is x-grid, Y is y-grid
        mesh_x, mesh_y = paddle.meshgrid(I_r_grid_x, I_r_grid_y)  # returns W x H each? (paddle doc: returns list)
        # paddle.meshgrid returns (W x H) so stack and transpose to get H x W x 2
        P = paddle.stack([mesh_x, mesh_y], axis=2)
        P = paddle.transpose(P, perm=[1, 0, 2])  # Ensure shape is H x W x 2
        return P.reshape([-1, 2])  # n x 2

    def build_inv_delta_C_paddle(self, C):
        """Return inv_delta_C which is needed to calculate T"""
        F = self.F
        hat_eye = paddle.eye(F, dtype="float32")  # F x F
        hat_C = (
            paddle.norm(C.reshape([1, F, 2]) - C.reshape([F, 1, 2]), axis=2) + hat_eye
        )
        hat_C = (hat_C**2) * paddle.log(hat_C)
        delta_C = paddle.concat(  # F+3 x F+3
            [
                paddle.concat(
                    [paddle.ones((F, 1), dtype="float32"), C, hat_C], axis=1
                ),  # F x F+3
                paddle.concat(
                    [
                        paddle.zeros((2, 3), dtype="float32"),
                        paddle.transpose(C, perm=[1, 0]),
                    ],
                    axis=1,
                ),  # 2 x F+3
                paddle.concat(
                    [
                        paddle.zeros((1, 3), dtype="float32"),
                        paddle.ones((1, F), dtype="float32"),
                    ],
                    axis=1,
                ),  # 1 x F+3
            ],
            axis=0,
        )
        inv_delta_C = paddle.inverse(delta_C)
        return inv_delta_C  # F+3 x F+3

    def build_P_hat_paddle(self, C, P):
        """
        P: n x 2 (float32/64 depending on caller)
        C: F x 2
        Return P_hat: n x (F+3)
        """
        F = self.F
        eps = self.eps
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        if n == 0:
            raise ValueError("Grid P is empty (n == 0). Check I_r_size.")

        # tile P to n x F x 2
        P_tile = paddle.tile(paddle.unsqueeze(P, axis=1), (1, F, 1))
        C_tile = paddle.unsqueeze(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        # rbf_norm: n x F
        rbf_norm = paddle.norm(P_diff, p=2, axis=2, keepdim=False)

        # rbf: n x F ; add eps before log to avoid log(0)
        rbf = paddle.multiply(paddle.square(rbf_norm), paddle.log(rbf_norm + eps))
        P_hat = paddle.concat([paddle.ones((n, 1), dtype=P.dtype), P, rbf], axis=1)
        return P_hat  # n x F+3

    def get_expand_tensor(self, batch_C_prime):
        B, H, C = batch_C_prime.shape
        batch_C_prime = batch_C_prime.reshape([B, H * C])
        batch_C_ex_part_tensor = self.fc(batch_C_prime)
        batch_C_ex_part_tensor = batch_C_ex_part_tensor.reshape([-1, 3, 2])
        return batch_C_ex_part_tensor


class TPS(nn.Layer):
    def __init__(self, in_channels, num_fiducial, loc_lr, model_name, min_hw=8):
        """
        min_hw: minimum height/width allowed; inputs smaller than this will be upsampled
        to avoid spatial collapse after pooling. Set to None to disable automatic upsampling.
        """
        super(TPS, self).__init__()
        self.loc_net = LocalizationNetwork(
            in_channels, num_fiducial, loc_lr, model_name
        )
        self.grid_generator = GridGenerator(self.loc_net.out_channels, num_fiducial)
        self.out_channels = in_channels
        self.min_hw = min_hw

    def _ensure_min_size(self, image):
        """
        If image H or W is below min_hw, upsample to (min_hw, min_hw) preserving channels.
        image: [B, C, H, W]
        """
        if self.min_hw is None:
            return image, False

        H = int(image.shape[2])
        W = int(image.shape[3])
        need_resize = (H < self.min_hw) or (W < self.min_hw)
        if not need_resize:
            return image, False

        target_h = max(H, self.min_hw)
        target_w = max(W, self.min_hw)
        # Use bilinear interpolation (align_corners=False)
        # Paddle expects size as tuple (H, W)
        image_resized = F.interpolate(image, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return image_resized, True

    def forward(self, image):
        """
        image: [B, C, H, W]
        returns: transformed image [B, C, H, W] (H/W possibly changed if min_hw triggered)
        """
        # Validate image dims
        if image is None:
            raise ValueError("TPS.forward received None image")

        if len(image.shape) != 4:
            raise ValueError(f"TPS.forward expects 4D tensor [B,C,H,W], got shape={image.shape}")

        H = int(image.shape[2])
        W = int(image.shape[3])
        if H <= 0 or W <= 0:
            raise ValueError(f"TPS.forward received image with non-positive H/W: H={H}, W={W}")

        # If tiny, upsample automatically to avoid pooling collapsing to zero dims
        image, resized = self._ensure_min_size(image)

        # Ensure gradients flow into loc_net
        image.stop_gradient = False
        batch_C_prime = self.loc_net(image)  # B x F x 2

        # Build grid for current image size
        I_r_size = (int(image.shape[2]), int(image.shape[3]))
        batch_P_prime = self.grid_generator(batch_C_prime, I_r_size)  # B x (H*W) x 2

        # reshape to [B, H, W, 2] for grid_sample
        B = batch_P_prime.shape[0]
        npoints = batch_P_prime.shape[1]
        expected_n = I_r_size[0] * I_r_size[1]
        if int(npoints) != expected_n:
            raise ValueError(f"Grid generator returned {npoints} points but expected {expected_n} for size {I_r_size}")

        batch_P_prime = batch_P_prime.reshape([-1, I_r_size[0], I_r_size[1], 2])

        # grid_sample expects grid in float32 normally
        is_fp16 = False
        if batch_P_prime.dtype != paddle.float32:
            data_type = batch_P_prime.dtype
            image = image.cast(paddle.float32)
            batch_P_prime = batch_P_prime.cast(paddle.float32)
            is_fp16 = True

        # sample
        batch_I_r = F.grid_sample(x=image, grid=batch_P_prime)

        if is_fp16:
            batch_I_r = batch_I_r.cast(data_type)

        return batch_I_r
