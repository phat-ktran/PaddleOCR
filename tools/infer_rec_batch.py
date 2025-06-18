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


def main():
    global_config = config["Global"]
    if config["Architecture"].get("algorithm") in [
        "UniMERNet",
        "PP-FormulaNet-S",
        "PP-FormulaNet-L",
        "PP-FormulaNet_plus-S",
        "PP-FormulaNet_plus-M",
        "PP-FormulaNet_plus-L",
    ]:
        config["PostProcess"]["is_infer"] = True
    # build post process
    post_process_class = build_post_process(config["PostProcess"], global_config)

    # build model
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in [
            "Distillation",
        ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                if (
                    config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead"
                ):  # multi head
                    out_channels_list = {}
                    if config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                        char_num = char_num - 2
                    if config["PostProcess"]["name"] == "DistillationNRTRLabelDecode":
                        char_num = char_num - 3
                    out_channels_list["CTCLabelDecode"] = char_num
                    out_channels_list["SARLabelDecode"] = char_num + 2
                    out_channels_list["NRTRLabelDecode"] = char_num + 3
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels_list"
                    ] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"]["out_channels"] = (
                        char_num
                    )
        elif config["Architecture"]["Head"]["name"] == "MultiHead":  # multi head
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

    if config["Architecture"].get("algorithm") in ["LaTeXOCR"]:
        config["Architecture"]["Backbone"]["is_predict"] = True
        config["Architecture"]["Backbone"]["is_export"] = True
        config["Architecture"]["Head"]["is_export"] = True

    model = build_model(config["Architecture"])

    load_model(config, model)

    # create data ops
    transforms = []
    for op in config["Eval"]["dataset"]["transforms"]:
        op_name = list(op)[0]
        if "Label" in op_name:
            continue
        elif op_name in ["RecResizeImg"]:
            op[op_name]["infer_mode"] = True
        elif op_name == "KeepKeys":
            if config["Architecture"]["algorithm"] == "SRN":
                op[op_name]["keep_keys"] = [
                    "image",
                    "encoder_word_pos",
                    "gsrm_word_pos",
                    "gsrm_slf_attn_bias1",
                    "gsrm_slf_attn_bias2",
                ]
            elif config["Architecture"]["algorithm"] == "SAR":
                op[op_name]["keep_keys"] = ["image", "valid_ratio"]
            elif config["Architecture"]["algorithm"] == "RobustScanner":
                op[op_name]["keep_keys"] = ["image", "valid_ratio", "word_positons"]
            else:
                op[op_name]["keep_keys"] = ["image"]
        transforms.append(op)
    global_config["infer_mode"] = True
    ops = create_operators(transforms, global_config)

    save_res_path = config["Global"].get(
        "save_res_path", "./output/rec/predicts_rec.txt"
    )
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.eval()

    infer_imgs = config["Global"]["infer_img"]
    infer_list = config["Global"].get("infer_list", None)

    # Get batch size from config
    batch_size = config["Eval"]["loader"].get("batch_size_per_card", 1)
    algorithm = config["Architecture"]["algorithm"]

    # Get all image files
    image_files = list(get_image_file_list(infer_imgs, infer_list=infer_list))
    logger.info(f"Processing {len(image_files)} images with batch size {batch_size}")

    with open(save_res_path, "w") as fout:
        for batch_start in range(0, len(image_files), batch_size):
            batch_end = min(batch_start + batch_size, len(image_files))
            batch_files = image_files[batch_start:batch_end]
            current_batch_size = len(batch_files)

            logger.info(
                f"Processing batch {batch_start // batch_size + 1}/{(len(image_files) + batch_size - 1) // batch_size}"
            )

            # Prepare batch data
            batch_images = []
            batch_data_list = []
            for file in batch_files:
                logger.info("Loading image: {}".format(file))
                with open(file, "rb") as f:
                    img = f.read()

                if algorithm in [
                    "UniMERNet",
                    "PP-FormulaNet-S",
                    "PP-FormulaNet-L",
                    "PP-FormulaNet_plus-S",
                    "PP-FormulaNet_plus-M",
                    "PP-FormulaNet_plus-L",
                ]:
                    data = {"image": img, "filename": file}
                else:
                    data = {"image": img}

                batch_result = transform(data, ops)
                batch_data_list.append(batch_result)
                batch_images.append(batch_result[0])

            # Stack images into batch tensor
            images_batch = np.stack(batch_images, axis=0)
            images_tensor = paddle.to_tensor(images_batch)

            if config["Architecture"]["algorithm"] == "SRN":
                encoder_word_pos_batch = np.stack(
                    [batch_data[1] for batch_data in batch_data_list], axis=0
                )
                gsrm_word_pos_batch = np.stack(
                    [batch_data[2] for batch_data in batch_data_list], axis=0
                )
                gsrm_slf_attn_bias1_batch = np.stack(
                    [batch_data[3] for batch_data in batch_data_list], axis=0
                )
                gsrm_slf_attn_bias2_batch = np.stack(
                    [batch_data[4] for batch_data in batch_data_list], axis=0
                )

                others_batch = [
                    paddle.to_tensor(encoder_word_pos_batch),
                    paddle.to_tensor(gsrm_word_pos_batch),
                    paddle.to_tensor(gsrm_slf_attn_bias1_batch),
                    paddle.to_tensor(gsrm_slf_attn_bias2_batch),
                ]
            if config["Architecture"]["algorithm"] == "SAR":
                valid_ratio_batch = np.stack(
                    [batch_data[-1] for batch_data in batch_data_list], axis=0
                )
                img_metas = [paddle.to_tensor(valid_ratio_batch)]
            if config["Architecture"]["algorithm"] == "RobustScanner":
                valid_ratio_batch = np.stack(
                    [batch_data[1] for batch_data in batch_data_list], axis=0
                )
                word_positions_batch = np.stack(
                    [batch_data[2] for batch_data in batch_data_list], axis=0
                )
                img_metas = [
                    paddle.to_tensor(valid_ratio_batch),
                    paddle.to_tensor(word_positions_batch),
                ]
            if config["Architecture"]["algorithm"] == "CAN":
                image_mask_batch = paddle.ones(images_tensor.shape, dtype="float32")
                label_batch = paddle.ones((current_batch_size, 36), dtype="int64")

            if config["Architecture"]["algorithm"] == "SRN":
                preds = model(images_tensor, others_batch)
            elif config["Architecture"]["algorithm"] == "SAR":
                preds = model(images_tensor, img_metas)
            elif config["Architecture"]["algorithm"] == "RobustScanner":
                preds = model(images_tensor, img_metas)
            elif config["Architecture"]["algorithm"] == "CAN":
                preds = model([images_tensor, image_mask_batch, label_batch])
            else:
                preds = model(images_tensor)

            # Process batch predictions
            post_results = post_process_class(preds)

            # Handle different post-processing result formats for batch
            for i, file in enumerate(batch_files):
                info = None

                if isinstance(post_results, dict):
                    # Handle dictionary results - extract for current batch item
                    rec_info = dict()
                    for key in post_results:
                        if (
                            len(post_results[key]) > i
                            and len(post_results[key][i]) >= 2
                        ):
                            rec_info[key] = {
                                "label": post_results[key][i][0],
                                "score": float(post_results[key][i][1]),
                            }
                    info = json.dumps(rec_info, ensure_ascii=False)

                elif isinstance(post_results, list):
                    if isinstance(post_results[0], int):
                        # For RFLearning CNT branch - handle batch
                        info = str(
                            post_results[i]
                            if i < len(post_results)
                            else post_results[0]
                        )
                    elif algorithm in [
                        "LaTeXOCR",
                        "UniMERNet",
                        "PP-FormulaNet-S",
                        "PP-FormulaNet-L",
                        "PP-FormulaNet_plus-S",
                        "PP-FormulaNet_plus-M",
                        "PP-FormulaNet_plus-L",
                    ]:
                        info = str(post_results[i] if i < len(post_results) else "")
                    else:
                        # Handle standard list format for batch
                        if i < len(post_results) and len(post_results[i]) >= 2:
                            info = post_results[i][0] + "\t" + str(post_results[i][1])
                        elif len(post_results) > 0 and len(post_results[0]) >= 2:
                            # Fallback to first result if batch indexing fails
                            info = post_results[0][0] + "\t" + str(post_results[0][1])

                if info is not None:
                    logger.info(f"\t result for {file}: {info}")
                    fout.write(file + "\t" + info + "\n")
                else:
                    logger.warning(f"No valid result for {file}")
    logger.info("success!")


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()
    main()
