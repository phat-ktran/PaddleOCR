import paddle
from paddle import nn

from .det_resnet_vd import ConvBNLayer

__all__ = ["NomNaNet"]


class NomNaNet(nn.Layer):
    def __init__(self, in_channels, channel_last=False, channels=[64, 128, 256, 512]):
        super().__init__()
        assert len(channels) == 4, "Only support 4 blocks."
        self.in_channels = in_channels
        self.channel_last = channel_last
        self.out_channels = channels[-1]
        self.block1 = nn.Sequential(
            ConvBNLayer(
                in_channels=in_channels,
                out_channels=channels[0],
                kernel_size=3,
                stride=1,
                act="relu",
            ),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
        )
        self.block2 = nn.Sequential(
            ConvBNLayer(
                in_channels=channels[0],
                out_channels=channels[1],
                kernel_size=3,
                stride=1,
                act="relu",
            ),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
        )
        self.block3 = nn.Sequential(
            ConvBNLayer(
                in_channels=channels[1],
                out_channels=channels[2],
                kernel_size=3,
                stride=1,
                act="relu",
            ),
            ConvBNLayer(
                in_channels=channels[2],
                out_channels=channels[2],
                kernel_size=3,
                stride=1,
                act="relu",
            ),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
        )
        self.block4 = nn.Sequential(
            ConvBNLayer(
                in_channels=channels[2],
                out_channels=channels[3],
                kernel_size=3,
                stride=1,
                act="relu",
            ),
            ConvBNLayer(
                in_channels=channels[3],
                out_channels=channels[3],
                kernel_size=3,
                stride=1,
                act="relu",
            ),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
        )
        self.block5 = nn.Sequential(
            ConvBNLayer(
                in_channels=channels[3],
                out_channels=channels[3],
                kernel_size=2,
                stride=1,
                padding=0,
                act="relu",
            ),
            ConvBNLayer(
                in_channels=channels[3],
                out_channels=channels[3],
                kernel_size=2,
                stride=1,
                padding=0,
                act="relu",
            ),
        )

    def forward(self, X: paddle.Tensor) -> paddle.Tensor:
        X = self.block1(X)
        X = self.block2(X)
        X = self.block3(X)
        X = self.block4(X)
        X = self.block5(X)
        if self.channel_last:
            X = paddle.transpose(X, [0, 3, 2, 1]) # N W H C
            X_shape = X.shape
            X = paddle.reshape(X, [X_shape[0], X_shape[1], X_shape[2] * X_shape[3]]) # N W C*H
        return X
