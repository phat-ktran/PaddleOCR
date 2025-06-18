# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import yaml
import paddle
from paddle.jit import to_static
from paddle.static import InputSpec

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in [".yml", ".yaml"], "only support yaml files for now"
    config = yaml.load(open(file_path, "rb"), Loader=yaml.Loader)
    return config


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument("-o", "--opt", nargs="+", help="set configuration options")
        self.add_argument(
            "-p",
            "--profiler_options",
            type=str,
            default=None,
            help="The option of profiler, which should be in format "
            '"key1=value1;key2=value2;key3=value3".',
        )
        self.add_argument(
            "--input_size",
            type=str,
            default=None,
            help="Specify the input size for the model in the format 'batch,channel,height,width'.",
        )
        self.add_argument(
            "--training",
            action="store_true",
            help="Enable training mode.",
        )

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        try:
            input_size = tuple(int(dim) for dim in args.input_size.split(","))
            assert len(input_size) == 4, "input_size must be in the format 'batch,channel,height,width'."
            args.input_size = input_size
        except ValueError:
            raise ValueError("input_size must contain only integers separated by commas.")
        assert args.config is not None, "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split("=")
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config


def main():
    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)
    global_config = config["Global"]

    # build post process
    post_process_class = build_post_process(config["PostProcess"], global_config)

    # Build model
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in [
            "Distillation",
        ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                if (
                    config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead"
                ):  # for multi head
                    if config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                        char_num = char_num - 2
                    if config["PostProcess"]["name"] == "DistillationNRTRLabelDecode":
                        char_num = char_num - 3
                    out_channels_list = {}
                    out_channels_list["CTCLabelDecode"] = char_num
                    # update SARLoss params
                    if (
                        list(config["Loss"]["loss_config_list"][-1].keys())[0]
                        == "DistillationSARLoss"
                    ):
                        config["Loss"]["loss_config_list"][-1]["DistillationSARLoss"][
                            "ignore_index"
                        ] = char_num + 1
                        out_channels_list["SARLabelDecode"] = char_num + 2
                    elif any(
                        "DistillationNRTRLoss" in d
                        for d in config["Loss"]["loss_config_list"]
                    ):
                        out_channels_list["NRTRLabelDecode"] = char_num + 3

                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels_list"
                    ] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"]["out_channels"] = (
                        char_num
                    )
        elif config["Architecture"]["Head"]["name"] == "MultiHead":  # for multi head
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            if config["PostProcess"]["name"] == "NRTRLabelDecode":
                char_num = char_num - 3
            out_channels_list = {}
            out_channels_list["CTCLabelDecode"] = char_num
            # update SARLoss params
            if list(config["Loss"]["loss_config_list"][1].keys())[0] == "SARLoss":
                if config["Loss"]["loss_config_list"][1]["SARLoss"] is None:
                    config["Loss"]["loss_config_list"][1]["SARLoss"] = {
                        "ignore_index": char_num + 1
                    }
                else:
                    config["Loss"]["loss_config_list"][1]["SARLoss"]["ignore_index"] = (
                        char_num + 1
                    )
                out_channels_list["SARLabelDecode"] = char_num + 2
            elif list(config["Loss"]["loss_config_list"][1].keys())[0] == "NRTRLoss":
                out_channels_list["NRTRLabelDecode"] = char_num + 3
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num

        if config["PostProcess"]["name"] == "SARLabelDecode":  # for SAR model
            config["Loss"]["ignore_index"] = char_num - 1

    model = build_model(config["Architecture"])

    if config["Global"].get("to_static", False):
        assert (
            "d2s_train_image_shape" in config["Global"]
        ), "d2s_train_image_shape must be assigned for static training mode..."
        supported_list = [
            "SVTR_LCNet",
            "SVTR",
            "SVTR_HGNet",
        ]
        if config["Architecture"]["algorithm"] in ["Distillation"]:
            algo = list(config["Architecture"]["Models"].values())[0]["algorithm"]
        else:
            algo = config["Architecture"]["algorithm"]
        assert (
            algo in supported_list
        ), f"algorithms that supports static training must in in {supported_list} but got {algo}"
    
        specs = [
            InputSpec([None] + config["Global"]["d2s_train_image_shape"], dtype="float32")
        ]
    
        if algo == "SVTR_LCNet":
            specs.append(
                [
                    InputSpec([None, config["Global"]["max_text_length"]], dtype="int64"),
                    InputSpec([None, config["Global"]["max_text_length"]], dtype="int64"),
                    InputSpec([None], dtype="int64"),
                    InputSpec([None], dtype="float64"),
                ]
            )
        elif algo == "SVTR":
            specs.append(
                [
                    InputSpec([None, config["Global"]["max_text_length"]], dtype="int64"),
                    InputSpec([None], dtype="int64"),
                ]
            )
        model = to_static(model, input_spec=specs)

    paddle.summary(model, input_size=FLAGS.input_size)


if __name__ == "__main__":
    main()
