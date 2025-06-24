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
from ppocr.utils.utility import get_image_file_list, map_to_json_schema
import tools.program as program


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


def _construct_model(config, return_all_feats=False):
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

    config["Architecture"]["Head"]["return_all_feats"] = return_all_feats
    model = build_model(config["Architecture"])

    load_model(config, model)    

    return model, post_process_class

def main():
    master_config = config["master"]
    master, master_post_process_class = _construct_model(master_config)
    master_ops = init_transforms(["image"], master_config["Global"])
    save_res_path = master_config["Global"].get(
        "save_res_path", "./output/rec/predicts_rec.txt"
    )
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))
    master.eval()
    
    slave_config = config["slave"]
    slave, slave_post_process_class = _construct_model(slave_config, return_all_feats=True)
    slave_ops = init_transforms(["image", "label_ctc", "label_gtc", "length"], slave_config["Global"], False)
    slave.eval()

    infer_imgs = master_config["Global"]["infer_img"]
    infer_list = master_config["Global"].get("infer_list", None)
    with open(save_res_path, "w") as fout:
        for file in get_image_file_list(infer_imgs, infer_list=infer_list):
            logger.info("infer_img: {}".format(file))
            with open(file, "rb") as f:
                img = f.read()
                data = {"image": img}
            batch = transform(data, master_ops)
            images = np.expand_dims(batch[0], axis=0)
            images = paddle.to_tensor(images)

            kwargs = prepare_args(master_config["Global"])
            preds = master.forward(images)
            post_result = master_post_process_class(preds, **kwargs)
            if len(post_result[0]) >= 2:
                post_result = {
                    "ctc": [
                        {
                            "text": post_result[0][0],
                            "confidence": post_result[0][1]
                        }
                    ]
                }
            else:
                post_result = map_to_json_schema(post_result)

            if len(post_result["ctc"][0]["text"]) > 0:
                with open(file, "rb") as f:
                    img = f.read()
                    data = {"image": img, "label": post_result["ctc"][0]["text"]}
                batch = transform(data, slave_ops)
                images = np.expand_dims(batch[0], axis=0)
                images = paddle.to_tensor(images)
                
                label_ctc = np.expand_dims(batch[1], axis=0)
                label_ctc = paddle.to_tensor(label_ctc)
    
                label_gtc = np.expand_dims(batch[2], axis=0)
                label_gtc = paddle.to_tensor(label_gtc)
    
                length = np.expand_dims(batch[3], axis=0)
                length = paddle.to_tensor(length)
    
                batch = (images, label_ctc, label_gtc, length)
                preds = slave.forward(images, batch[1:])
                slave_post_result = slave_post_process_class(preds, **kwargs)
                slave_post_result = map_to_json_schema(post_result)
                post_result["gtc"] = slave_post_result["gtc"]
                
            info = json.dumps(post_result, ensure_ascii=False)
            if info is not None:
                logger.info("\t result: {}".format(info))
                fout.write(file + "\t" + info + "\n")

    logger.info("success!")


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()
    main()
