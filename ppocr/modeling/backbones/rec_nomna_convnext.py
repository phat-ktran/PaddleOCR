import paddle
from paddle import ParamAttr, nn
from paddle.nn.initializer.kaiming import KaimingNormal

__all__ = [ "ConvNeXt" ]

class Stem(nn.Layer):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        act,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            data_format=ConvNeXt.DATA_FORMAT,
            weight_attr=ParamAttr(initializer=KaimingNormal())
        )
        self.normalize = nn.LayerNorm([out_channels])

    def forward(self, X: paddle.Tensor) -> paddle.Tensor:
        return self.normalize(self.conv(X))

class InvertedBottleneckUnit(nn.Layer):
    def __init__(self,
        in_channels,
        alpha,
        kernel_size,
        stride,
        padding,
        act,
    ):
        super().__init__()
        self.expanded = in_channels * alpha
        self.dense = nn.Conv2D(
            in_channels,
            self.expanded,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            data_format=ConvNeXt.DATA_FORMAT,
            weight_attr=ParamAttr(initializer=KaimingNormal())
        )
        self.normalize = nn.LayerNorm([self.expanded])
        self.conv1x1_1 = nn.Conv2D(
            self.expanded,
            self.expanded,
            kernel_size=1,
            stride=1,
            padding=0,
            data_format=ConvNeXt.DATA_FORMAT,
            weight_attr=ParamAttr(initializer=KaimingNormal())
        )
        self.act = nn.ReLU() if act == "relu" else nn.GELU()
        self.conv1x1_2 = nn.Conv2D(
            self.expanded,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            data_format=ConvNeXt.DATA_FORMAT,
            weight_attr=ParamAttr(initializer=KaimingNormal())
        )

    def forward(self, X: paddle.Tensor) -> paddle.Tensor:
        return X + self.conv1x1_2(self.act(self.conv1x1_1(self.normalize(self.dense(X)))))

class TransitionUnit(nn.Layer):
    def __init__(self,
        in_channels,
        kernel_size,
        stride,
        padding,
        act,
    ):
        super().__init__()
        self.normalize = nn.LayerNorm([in_channels])
        self.identity = nn.Conv2D(
            in_channels,
            in_channels * ConvNeXt.EXPANSION,
            kernel_size=1,
            stride=ConvNeXt.EXPANSION,
            padding=0,
            data_format=ConvNeXt.DATA_FORMAT,
            weight_attr=ParamAttr(initializer=KaimingNormal())
        )
        self.residual = nn.Conv2D(
            in_channels,
            in_channels * ConvNeXt.EXPANSION,
            kernel_size=kernel_size,
            stride=ConvNeXt.EXPANSION,
            padding=padding,
            data_format=ConvNeXt.DATA_FORMAT,
            weight_attr=ParamAttr(initializer=KaimingNormal())
        )

    def forward(self, X: paddle.Tensor) -> paddle.Tensor:
        X = self.normalize(X)
        return self.identity(X) + self.residual(X)

def _make_transition(in_channels, transition_config, act) -> TransitionUnit:
    return TransitionUnit(
        in_channels,
        kernel_size=transition_config.get("kernel_size", 3),
        stride=transition_config.get("stride", 2),
        padding=transition_config.get("padding", 1),
        act=act
    )

def _make_bottleneck(in_channels, bottleneck_config, act) -> InvertedBottleneckUnit:
    return InvertedBottleneckUnit(
        in_channels,
        bottleneck_config["alpha"],
        kernel_size=bottleneck_config.get("kernel_size", 7),
        stride=bottleneck_config.get("stride", 1),
        padding=bottleneck_config.get("padding", 3),
        act=act
    )

def _make_stem(in_channels, stem_config, act) -> Stem:
    return Stem(
        in_channels,
        stem_config["out_channels"],
        kernel_size=stem_config.get("kernel_size", 4),
        stride=stem_config.get("stride", 4),
        padding=stem_config.get("padding", 0),
        act=act
    )

class ConvNeXt(nn.Layer):
    DATA_FORMAT = "NHWC"
    EXPANSION = 2
    def __init__(
        self,
        in_channels,
        stages,
        act = "gelu"
    ):
        super().__init__()
        assert len(stages) == 4, "Only support 5 stages in ConvNeXt."
        assert act in ["gelu", "relu"], "Only support ReLU and GELU."
        self.in_channels = in_channels
        self.act = act
        self.stem_config ={
            "out_channels": 64,
            "kernel_size": (2,4),
            "stride": (2,4),
            "padding": 0
        },
        self.bottleneck_config={
            "kernel_size": 7,
            "alpha": 4,
            "stride": 1,
            "padding": 3
        },
        self.transition_config={
            "kernel_size": 3,
            "stride": 2,
            "padding": 1
        }
        # Stage 1
        self.stem = _make_stem(self.in_channels, self.stem_config, self.act)
        out_channels = self.stem.out_channels
        # Stage 2
        self.stage2 = nn.LayerList([_make_bottleneck(self.stem.out_channels, self.bottleneck_config, self.act) for _ in range(stages[0])])
        # Stage 3
        self.stage3 = nn.LayerList([_make_transition(out_channels, self.transition_config, self.act)])
        out_channels *= self.EXPANSION
        self.stage3.extend([_make_bottleneck(out_channels, self.bottleneck_config, self.act) for _ in range(stages[1]-1)])
        # Stage 4
        self.stage4 = nn.LayerList([_make_transition(out_channels, self.transition_config, self.act)])
        out_channels *= self.EXPANSION
        self.stage4.extend([_make_bottleneck(out_channels, self.bottleneck_config, self.act) for _ in range(stages[2]-1)])
        # Stage 5
        self.stage5 = nn.LayerList([_make_transition(out_channels, self.transition_config, self.act)])
        out_channels *= self.EXPANSION
        self.stage5.extend([_make_bottleneck(out_channels, self.bottleneck_config, self.act) for _ in range(stages[3]-1)])

    def forward(self, X: paddle.Tensor) -> paddle.Tensor:
        S = self.stem(X.transpose([0,2,3,1]))
        for layer in self.stage2:
            S = layer(S)
        for layer in self.stage3:
            S = layer(S)
        for layer in self.stage4:
            S = layer(S)
        for layer in self.stage5:
            S = layer(S)
        return S.transpose([0, 3, 2, 1]) # N, C, 1, S

if __name__ == '__main__':
    model = ConvNeXt(
        in_channels=3,
        stages=[3,3,6,3],
    )
    paddle.summary(model, input_size=(3, 432, 32, 3))
