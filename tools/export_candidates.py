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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys
import json

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program


def map_to_json_schema(data):
    """
    Maps input data to the specified JSON schema, converting NumPy types to Python types
    for JSON serializability and renaming fields for clarity.

    Args:
        data (dict): Input data with 'ctc' and 'gtc' keys, as produced by NRTRLabelDecode.

    Returns:
        dict: Data mapped to the schema with 'ctc' and 'gtc' containing lists of dictionaries.
    """

    def convert_types(item):
        """Recursively converts NumPy types to Python types."""
        if isinstance(item, dict):
            return {key: convert_types(value) for key, value in item.items()}
        if isinstance(item, list):
            return [convert_types(element) for element in item]
        if isinstance(item, tuple):
            # Handle tuples like (char, prob) in top_k or (text, conf, candidates)
            return tuple(convert_types(element) for element in item)
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return float(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        return item

    result = {}

    # Map 'ctc' to list of {'text': str, 'confidence': float}
    if "ctc" in data:
        result["ctc"] = [
            {"text": convert_types(item[0]), "confidence": convert_types(item[1])}
            for item in data["ctc"]
        ]

    # Map 'gtc' to list of {'texts': str, 'confidence': float, 'top_k': list}
    if "gtc" in data:
        result["gtc"] = [
            {
                "text": convert_types(item[0]),
                "confidence": convert_types(item[1]),
                "top_k": convert_types(item[2]) if len(item) > 2 else [],
            }
            for item in data["gtc"]
        ]

    return result


def prepare_args(global_config):
    kwargs = dict()
    if config["PostProcess"]["name"] == "BeamCTCLabelDecode":
        kwargs["use_beam_search"] = global_config.get("use_beam_search", False)
        kwargs["beam_width"] = global_config.get("beam_width", 5)
        kwargs["return_all_beams"] = global_config.get("return_all_beams", False)
    elif config["PostProcess"]["name"] == "MultiHeadLabelDecode":
        if "BeamCTCLabelDecode" in config["PostProcess"]["decoder_list"]:
            kwargs["ctc"] = {
                "use_beam_search": global_config.get("use_beam_search", False),
                "beam_width": global_config.get("beam_width", False),
                "return_all_beams": global_config.get("return_all_beams", False),
            }
        kwargs["gtc"] = {}

    return kwargs


def init_transforms(keys, global_config, ignore_encode=True):
    transforms = []
    for op in config["Eval"]["dataset"]["transforms"]:
        op_name = list(op)[0]
        if "Label" in op_name and ignore_encode:
            continue
        elif op_name in ["RecResizeImg"]:
            op[op_name]["infer_mode"] = True
        elif op_name == "KeepKeys":
            op[op_name]["keep_keys"] = keys
        transforms.append(op)
    global_config["infer_mode"] = True
    ops = create_operators(transforms, global_config)
    return ops


def main():
    global_config = config["Global"]
    # build post process
    post_process_class = build_post_process(config["PostProcess"], global_config)

    # build model
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["Head"]["name"] == "MultiHead":  # multi head
            out_channels_list = {}
            char_num = len(getattr(post_process_class, "character"))
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            if config["PostProcess"]["name"] == "NRTRLabelDecode":
                char_num = char_num - 3
            out_channels_list["CTCLabelDecode"] = char_num
            out_channels_list["SARLabelDecode"] = char_num + 2
            out_channels_list["NRTRLabelDecode"] = char_num + 3
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num

    config["Architecture"]["Head"]["return_all_feats"] = True
    model = build_model(config["Architecture"])

    load_model(config, model)

    # create data ops
    ops = init_transforms(["image"], global_config)

    save_res_path = config["Global"].get(
        "save_res_path", "./output/rec/predicts_rec.txt"
    )
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.eval()

    infer_imgs = config["Global"]["infer_img"]
    infer_list = config["Global"].get("infer_list", None)
    with open(save_res_path, "w") as fout:
        for file in get_image_file_list(infer_imgs, infer_list=infer_list):
            logger.info("infer_img: {}".format(file))
            with open(file, "rb") as f:
                img = f.read()
                data = {"image": img}
            batch = transform(data, ops)
            images = np.expand_dims(batch[0], axis=0)
            images = paddle.to_tensor(images)

            kwargs = prepare_args(global_config)

            preds = model.forward(images)
            post_result = post_process_class(preds, **kwargs)
            post_result = map_to_json_schema(post_result)

            ops = init_transforms(
                ["image", "label_ctc", "label_gtc", "length"], global_config, False
            )
            data = {"image": img, "label": post_result["ctc"][0]["text"]}
            batch = transform(data, ops)
            images = np.expand_dims(batch[0], axis=0)
            images = paddle.to_tensor(images)

            preds = model.forward(images, batch[1:])
            post_result = post_process_class(preds, **kwargs)

            info = json.dumps(map_to_json_schema(post_result), ensure_ascii=False)
            if info is not None:
                logger.info("\t result: {}".format(info))
                fout.write(file + "\t" + info + "\n")

            ops = init_transforms(["image"], global_config)
    logger.info("success!")


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()
    main()
