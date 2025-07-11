# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.nn as nn

from ppocr.modeling.necks.rnn import (
    Im2Seq,
    SequenceEncoder,
    trunc_normal_,
    zeros_,
)
from .rec_nrtr_head import Transformer
from .rec_ctc_head import CTCHead
from .rec_sar_head import SARHead


class FCTranspose(nn.Layer):
    def __init__(self, in_channels, out_channels, only_transpose=False):
        super().__init__()
        self.only_transpose = only_transpose
        if not self.only_transpose:
            self.fc = nn.Linear(in_channels, out_channels, bias_attr=False)

    def forward(self, x):
        if self.only_transpose:
            return x.transpose([0, 2, 1])
        else:
            return self.fc(x.transpose([0, 2, 1]))


class AddPos(nn.Layer):
    def __init__(self, dim, w):
        super().__init__()
        self.dec_pos_embed = self.create_parameter(
            shape=[1, w, dim], default_initializer=zeros_
        )
        self.add_parameter("dec_pos_embed", self.dec_pos_embed)
        trunc_normal_(self.dec_pos_embed)

    def forward(self, x):
        x = x + self.dec_pos_embed[:, : x.shape[1], :]
        return x


class MultiHead(nn.Layer):
    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__()
        self.head_list = kwargs.pop("head_list")
        self.use_pool = kwargs.get("use_pool", False)
        self.use_pos = kwargs.get("use_pos", False)
        self.return_all_feats = kwargs.get("return_all_feats", False)
        self.in_channels = in_channels
        self.return_candidates_per_timestep = kwargs.get("return_candidates_per_timestep", False)
        self.k = kwargs.get("k", None)
        if self.use_pool:
            self.pool_kernel_size = kwargs.get("pool_kernel_size", [3, 2])
            self.pool_stride = kwargs.get("pool_stride", self.pool_kernel_size)
            self.pool = nn.AvgPool2D(
                kernel_size=self.pool_kernel_size, stride=self.pool_stride, padding=0
            )
        self.gtc_head = "sar"
        assert len(self.head_list) >= 2
        for idx, head_name in enumerate(self.head_list):
            name = list(head_name)[0]
            if name == "SARHead":
                # sar head
                sar_args = self.head_list[idx][name]
                self.sar_head = eval(name)(
                    in_channels=in_channels,
                    out_channels=out_channels_list["SARLabelDecode"],
                    **sar_args,
                )
            elif name == "NRTRHead":
                gtc_args = self.head_list[idx][name]
                max_text_length = gtc_args.get("max_text_length", 25)
                nrtr_dim = gtc_args.get("nrtr_dim", 256)
                num_decoder_layers = gtc_args.get("num_decoder_layers", 4)
                if self.use_pos:
                    self.before_gtc = nn.Sequential(
                        nn.Flatten(2),
                        FCTranspose(in_channels, nrtr_dim),
                        AddPos(nrtr_dim, 80),
                    )
                else:
                    self.before_gtc = nn.Sequential(
                        nn.Flatten(2), FCTranspose(in_channels, nrtr_dim)
                    )

                self.gtc_head = Transformer(
                    d_model=nrtr_dim,
                    nhead=nrtr_dim // 32,
                    num_encoder_layers=-1,
                    beam_size=-1,
                    num_decoder_layers=num_decoder_layers,
                    max_len=max_text_length,
                    dim_feedforward=nrtr_dim * 4,
                    out_channels=out_channels_list["NRTRLabelDecode"],
                )
            elif name == "CTCHead":
                # ctc neck
                neck_args = self.head_list[idx][name].get("Neck", None)
                if neck_args:
                    self.encoder_reshape = Im2Seq(in_channels)
                    encoder_type = neck_args.pop("name")
                    self.ctc_encoder = SequenceEncoder(
                        in_channels=in_channels, encoder_type=encoder_type, **neck_args
                    )
                else:
                    self.ctc_encoder = None
                    self.encoder_reshape = None
                # ctc head
                head_args = self.head_list[idx][name]["Head"]
                self.ctc_head = eval(name)(
                    in_channels=self.ctc_encoder.out_channels
                    if neck_args
                    else in_channels,
                    out_channels=out_channels_list["CTCLabelDecode"],
                    **head_args,
                )
            else:
                raise NotImplementedError(
                    "{} is not supported in MultiHead yet".format(name)
                )

    def forward(self, x, targets=None, **kwargs):
        if self.use_pool:
            x = self.pool(
                x.reshape(
                    [0, self.pool_kernel_size[0], -1, self.in_channels]
                ).transpose([0, 3, 1, 2])
            )
        if self.ctc_encoder:
            ctc_encoder = self.ctc_encoder(x)
        else: 
            ctc_encoder = x
        ctc_out = self.ctc_head(ctc_encoder, targets)
        head_out = dict()
        head_out["ctc"] = ctc_out
        head_out["ctc_neck"] = ctc_encoder
        if self.return_all_feats or self.training:
            if self.gtc_head == "sar":
                sar_out = self.sar_head(x, targets[1:])
                head_out["sar"] = sar_out
            else:
                targets_arg = targets[1:] if targets else None
                gtc_out = self.gtc_head(self.before_gtc(x), targets_arg, self.return_candidates_per_timestep, self.k)
                head_out["gtc"] = gtc_out
        # eval mode for single ctc decoding
        if not self.training and not self.return_all_feats:
            return ctc_out
        # training and other decoding strategies
        return head_out
