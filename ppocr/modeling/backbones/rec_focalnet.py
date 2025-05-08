# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn.initializer import KaimingNormal
from paddle.nn.initializer import TruncatedNormal, Constant, Normal

trunc_normal_ = TruncatedNormal(std=0.02)
normal_ = Normal
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias_attr=False,
        groups=1,
        act=nn.GELU,
    ):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()),
            bias_attr=bias_attr,
        )
        self.norm = nn.BatchNorm2D(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Mlp(nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FocalModulation(nn.Layer):
    def __init__(
        self,
        in_channels,
        focal_window,
        focal_level,
        act=nn.GELU,
        focal_factor=2,
        proj_drop=0.1,
        apply_norm_after=False,
        normalize_modulator=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = self.in_channels

        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.apply_norm_after = apply_norm_after
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(
            self.in_channels, 2 * self.in_channels + (self.focal_level + 1)
        )
        self.h = ConvBNLayer(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act=act,
            bias_attr=None,
        )

        self.act = act()
        self.proj = nn.Linear(self.in_channels, self.out_channels)
        self.focal_layers = nn.LayerList()
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                ConvBNLayer(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    groups=self.in_channels,
                    padding=kernel_size // 2,
                    act=act,
                    bias_attr=None,
                )
            )
            self.kernel_sizes.append(kernel_size)
        if self.apply_norm_after:
            self.ln = nn.LayerNorm(self.out_channels)
        self.proj_drop = nn.Dropout(p=proj_drop, mode="downscale_in_infer")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[-1]

        # pre linear projection
        x = self.f(x).transpose((0, 3, 1, 2)).contiguous()
        q, ctx, self.gates = paddle.split(x, (C, C, self.focal_level + 1), 1)

        # context aggreation
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * self.gates[:, l : l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * self.gates[:, self.focal_level :]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        self.modulator = self.h(ctx_all)
        x_out = q * self.modulator
        x_out = x_out.transpose((0, 2, 3, 1)).contiguous()
        if self.apply_norm_after:
            x_out = self.ln(x_out)

        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


class FocalNetBlock(nn.Layer):
    def __init__(
        self,
        in_channels,
        img_size,
        mlp_ratio=4.0,
        drop=0.1,
        drop_path=0.1,
        act_layer=nn.GELU,
        norm_layer="nn.LayerNorm",
        epsilon=1e-6,
        focal_level=1,
        focal_window=3,
        use_layerscale=False,
        layerscale_value=1e-4,
        use_postln=False,
        apply_norm_after=False,
        normalize_modulator=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.mlp_ratio = mlp_ratio

        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_postln = use_postln

        self.norm1 = eval(norm_layer)(in_channels, epsilon=epsilon)
        self.modulation = FocalModulation(
            in_channels,
            proj_drop=drop,
            focal_window=focal_window,
            focal_level=self.focal_level,
            apply_norm_after=apply_norm_after,
            normalize_modulator=normalize_modulator,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = eval(norm_layer)(in_channels, epsilon=epsilon)
        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=in_channels,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(
                layerscale_value * paddle.ones((in_channels)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                layerscale_value * paddle.ones((in_channels)), requires_grad=True
            )

        self.H = img_size[0]
        self.W = img_size[1]

    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape
        shortcut = x

        # Focal Modulation
        x = x if self.use_postln else self.norm1(x)
        x = x.view((B, H, W, C))
        x = self.modulation(x).view((B, H * W, C))
        x = x if not self.use_postln else self.norm1(x)

        # FFN
        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(
            self.gamma_2
            * (self.norm2(self.mlp(x)) if self.use_postln else self.mlp(self.norm2(x)))
        )

        return x


class BasicLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        img_size,
        depth,
        mlp_ratio=4.0,
        drop=0.1,
        drop_path=0.1,
        norm_layer="nn.LayerNorm",
        epsilon=1e-6,
        downsample=None,
        stride=[2, 1],
        focal_level=1,
        focal_window=1,
        use_layerscale=False,
        layerscale_value=1e-4,
        use_postln=False,
        apply_norm_after=False,
        normalize_modulator=False,
    ):
        super().__init__()
        self.dim = in_channels
        self.img_size = img_size
        self.depth = depth

        # build blocks
        self.blocks = nn.LayerList(
            [
                FocalNetBlock(
                    in_channels=in_channels,
                    img_size=img_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    focal_level=focal_level,
                    focal_window=focal_window,
                    use_layerscale=use_layerscale,
                    layerscale_value=layerscale_value,
                    use_postln=use_postln,
                    apply_norm_after=apply_norm_after,
                    normalize_modulator=normalize_modulator,
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample = downsample(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                sub_norm=norm_layer,
            )
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x)

        if self.downsample is not None:
            x = x.transpose((0, 2, 1)).reshape((x.shape[0], -1, H, W))
            x, Ho, Wo = self.downsample(x)
        else:
            Ho, Wo = H, W
        return x, Ho, Wo


class SubSample(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=[2, 1],
        sub_norm="nn.LayerNorm",
        act=None,
    ):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
        )
        self.norm = eval(sub_norm)(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        H, W = x.shape[2:]
        out = x.flatten(2).transpose((0, 2, 1))
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out, H, W


class PatchEmbed(nn.Layer):
    def __init__(
        self,
        img_size=[32, 640],
        in_channels=3,
        embed_dim=768,
        sub_num=2,
        norm_layer=None,
        patch_size=[4, 8],
        mode="pope",
    ):
        super().__init__()
        num_patches = (img_size[1] // 4) * (img_size[0] // 4)
        self.img_size = img_size
        self.num_patches = num_patches
        self.patches_resolution = [
            img_size[0] // 4,
            img_size[1] // 4,
        ]
        self.embed_dim = embed_dim
        self.norm = None
        if norm_layer:
            self.norm = eval(norm_layer)(embed_dim)
        if mode == "pope":
            if sub_num == 2:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None,
                    ),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None,
                    ),
                )
            if sub_num == 3:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 4,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None,
                    ),
                    ConvBNLayer(
                        in_channels=embed_dim // 4,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None,
                    ),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None,
                    ),
                )
        elif mode == "linear":
            self.proj = nn.Conv2D(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            )
            self.num_patches = (
                img_size[0] // patch_size[0] * img_size[1] // patch_size[1]
            )
            self.patches_resolution = [
                img_size[0] // patch_size[0],
                img_size[1] // patch_size[1],
            ]

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)
        H, W = x.shape[2:]
        x = x.flatten(2).transpose((0, 2, 1))
        if self.norm:
            x = self.norm(x)
        return x, H, W


class FocalNet(nn.Layer):
    def __init__(
        self,
        in_channels=3,
        embed_dim=96,
        img_size=[32, 640],
        patch_size=[4, 8],
        depths=[2, 2, 6, 2],
        focal_levels=[2, 2, 2, 2],
        focal_windows=[3, 3, 3, 3],
        subsample_stride=[2, 1],
        mlp_ratio=4.0,
        drop_rate=0.0,
        last_drop=0.1,
        drop_path_rate=0.1,
        epsilon=1e-6,
        out_char_num=25,
        norm_layer="nn.LayerNorm",
        last_stage=True,
        patch_norm=True,
        use_layerscale=False,
        layerscale_value=1e-4,
        use_postln=False,
        apply_norm_after=False,
        normalize_modulator=False,
        **kwargs,
    ):
        super().__init__()

        self.num_layers = len(depths)
        embed_dim = [embed_dim * (2**i) for i in range(self.num_layers)]

        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_channels = embed_dim[-1]
        self.mlp_ratio = mlp_ratio

        # split image into patches using either non-overlapped embedding or overlapped embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim[0],
            patch_size=patch_size,
            norm_layer=norm_layer if self.patch_norm else None,
            mode=kwargs.get("patch_mode", "linear"),
            sub_num=kwargs.get("sub_num", 2),
        )

        self.HW = self.patch_embed.patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.LayerList()
        self.h_factor = (
            subsample_stride
            if isinstance(subsample_stride, int)
            else subsample_stride[0]
        )
        self.w_factor = (
            subsample_stride
            if isinstance(subsample_stride, int)
            else subsample_stride[1]
        )
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                in_channels=embed_dim[i_layer],
                out_channels=embed_dim[i_layer + 1]
                if (i_layer < self.num_layers - 1)
                else None,
                img_size=(
                    self.HW[0] // (self.h_factor**i_layer),
                    self.HW[1] // (self.w_factor**i_layer),
                ),
                depth=depths[i_layer],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=SubSample if (i_layer < self.num_layers - 1) else None,
                stride=subsample_stride,
                focal_level=focal_levels[i_layer],
                focal_window=focal_windows[i_layer],
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                use_postln=use_postln,
                apply_norm_after=apply_norm_after,
                normalize_modulator=normalize_modulator,
            )
            self.layers.append(layer)

        self.last_stage = last_stage
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2D([1, out_char_num])
            self.last_conv = nn.Conv2D(
                in_channels=embed_dim[-1],
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False,
            )
            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(p=last_drop, mode="downscale_in_infer")

        self.use_postln = use_postln
        if not self.use_postln:
            self.norm = eval(norm_layer)(embed_dim[-1], epsilon=epsilon)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)

        if not self.use_postln:
            x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x) # B, L, C
        if self.last_stage:
            _, L, _ = x.shape
            h = self.HW[0] // self.h_factor ** (self.num_layers - 1)
            w = self.HW[1] // self.w_factor ** (self.num_layers - 1)
            assert h * w == L, (
                f"Mismatch shape, h = {h}, w = {w}, L = {L}, HW={self.HW}"
            )
            x = self.avg_pool(
                x.transpose([0, 2, 1]).reshape([0, self.embed_dim[-1], h, w])
            )
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
        return x
