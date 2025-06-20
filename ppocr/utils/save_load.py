# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import errno
import os
import pickle
import json
from packaging import version

from huggingface_hub import HfApi

import paddle
import paddle.nn.functional as F
from ppocr.utils.logging import get_logger
from ppocr.utils.network import maybe_download_params

try:
    import encryption  # Attempt to import the encryption module for AIStudio's encryption model

    encrypted = encryption.is_encryption_needed()
except ImportError:
    get_logger().warning("Skipping import of the encryption module.")
    encrypted = False  # Encryption is not needed if the module cannot be imported

__all__ = ["load_model"]


# just to determine the inference model file format
def get_FLAGS_json_format_model():
    # json format by default
    return os.environ.get("FLAGS_json_format_model", "1").lower() in ("1", "true", "t")


FLAGS_json_format_model = get_FLAGS_json_format_model()


def _mkdir_if_not_exist(path, logger):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    "be happy if some process has already created {}".format(path)
                )
            else:
                raise OSError("Failed to mkdir {}".format(path))


def load_model(config, model, optimizer=None, model_type="det"):
    """
    load model from checkpoint or pretrained_model
    """
    logger = get_logger()
    global_config = config["Global"]
    checkpoints = global_config.get("checkpoints")
    pretrained_model = global_config.get("pretrained_model")
    handle_mismatch = global_config.get("handle_mismatch", False)
    best_model_dict = {}
    is_float16 = False
    is_nlp_model = model_type == "kie" and config["Architecture"]["algorithm"] not in [
        "SDMGR"
    ]

    if is_nlp_model is True:
        # NOTE: for kie model dsitillation, resume training is not supported now
        if config["Architecture"]["algorithm"] in ["Distillation"]:
            return best_model_dict
        checkpoints = config["Architecture"]["Backbone"]["checkpoints"]
        # load kie method metric
        if checkpoints:
            if os.path.exists(os.path.join(checkpoints, "metric.states")):
                with open(os.path.join(checkpoints, "metric.states"), "rb") as f:
                    states_dict = pickle.load(f, encoding="latin1")
                best_model_dict = states_dict.get("best_model_dict", {})
                if "epoch" in states_dict:
                    best_model_dict["start_epoch"] = states_dict["epoch"] + 1
            logger.info("resume from {}".format(checkpoints))

            if optimizer is not None:
                if checkpoints[-1] in ["/", "\\"]:
                    checkpoints = checkpoints[:-1]
                if os.path.exists(checkpoints + ".pdopt"):
                    optim_dict = paddle.load(checkpoints + ".pdopt")
                    optimizer.set_state_dict(optim_dict)
                else:
                    logger.warning(
                        "{}.pdopt is not exists, params of optimizer is not loaded".format(
                            checkpoints
                        )
                    )

        return best_model_dict

    if checkpoints:
        if checkpoints.endswith(".pdparams"):
            checkpoints = checkpoints.replace(".pdparams", "")
        assert os.path.exists(checkpoints + ".pdparams"), (
            "The {}.pdparams does not exists!".format(checkpoints)
        )

        # load params from trained model
        params = paddle.load(checkpoints + ".pdparams")
        state_dict = model.state_dict()
        new_state_dict = {}
        for key, value in state_dict.items():
            if key not in params:
                logger.warning(
                    "{} not in loaded params {} !".format(key, params.keys())
                )
                continue
            pre_value = params[key]
            if pre_value.dtype == paddle.float16:
                is_float16 = True
            if pre_value.dtype != value.dtype:
                pre_value = pre_value.astype(value.dtype)
            if list(value.shape) == list(pre_value.shape):
                new_state_dict[key] = pre_value
            else:
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params shape {} !".format(
                        key, value.shape, pre_value.shape
                    )
                )
        model.set_state_dict(new_state_dict)
        if is_float16:
            logger.info(
                "The parameter type is float16, which is converted to float32 when loading"
            )
        if optimizer is not None:
            if os.path.exists(checkpoints + ".pdopt"):
                optim_dict = paddle.load(checkpoints + ".pdopt")
                optimizer.set_state_dict(optim_dict)
            else:
                logger.warning(
                    "{}.pdopt is not exists, params of optimizer is not loaded".format(
                        checkpoints
                    )
                )

        if os.path.exists(checkpoints + ".states"):
            with open(checkpoints + ".states", "rb") as f:
                states_dict = pickle.load(f, encoding="latin1")
            best_model_dict = states_dict.get("best_model_dict", {})
            best_model_dict["acc"] = 0.0
            if "epoch" in states_dict:
                best_model_dict["start_epoch"] = states_dict["epoch"] + 1
            if "global_step" in states_dict:
                best_model_dict["global_step"] = states_dict["global_step"]
        logger.info("resume from {}".format(checkpoints))
    elif pretrained_model:
        is_float16 = load_pretrained_params(model, pretrained_model, config)
    else:
        logger.info("train from scratch")
    best_model_dict["is_float16"] = is_float16
    return best_model_dict


def load_pretrained_params(model, path, config):
    override_head, top_k = False, None
    if "Global" in config:
        override_head = config["Global"].get("override_head", False)
        top_k = config["Global"].get("top_k", None)
    mapped_key_prefixes = config.get("mapped_key_prefixes", None)
    logger = get_logger()
    path = maybe_download_params(path)
    if path.endswith(".pdparams"):
        path = path.replace(".pdparams", "")
    assert os.path.exists(path + ".pdparams"), (
        "The {}.pdparams does not exists!".format(path)
    )

    params = paddle.load(path + ".pdparams")

    state_dict = model.state_dict()
    new_state_dict = {}
    is_float16 = False
    for k1 in params.keys():
        not_found = k1 not in state_dict.keys()
        k1_mapped = k1
        
        if mapped_key_prefixes and not_found:
            # Check if k1 has any of the supported prefixes
            for prefix in mapped_key_prefixes:
                if k1.startswith(prefix):
                    # Try mapping to a different prefix
                    mapped_prefix = mapped_key_prefixes[prefix]
                    k1_mapped = k1.replace(prefix, mapped_prefix, 1)
                    not_found = k1_mapped not in state_dict
                    if not not_found:
                        break
            
        if not_found:
            logger.warning("The pretrained params {} not in model".format(k1_mapped))
        else:
            if params[k1].dtype == paddle.float16:
                is_float16 = True
            if params[k1].dtype != state_dict[k1_mapped].dtype:
                params[k1_mapped] = params[k1].astype(state_dict[k1_mapped].dtype)
            if list(state_dict[k1_mapped].shape) == list(params[k1].shape):
                new_state_dict[k1_mapped] = params[k1]
            elif override_head:
                logger.warning(
                    "The shape of pretrained param {} {} does not match model param {} {}. Will try to copy overlapping dims.".format(
                        k1, params[k1].shape, k1, state_dict[k1_mapped].shape
                    )
                )
                
                if k1 == "backbone.pos_embed":                    
                    def resize_pos_embed(pretrained_pos_embed, target_shape):
                        """
                        Resize pretrained positional embeddings to match the target shape.
                        
                        Args:
                            pretrained_pos_embed (Tensor): Pretrained positional embeddings [1, num_patches_old, embed_dim].
                            target_shape (tuple): Desired shape [1, num_patches_new, embed_dim].
                        
                        Returns:
                            Tensor: Resized positional embeddings.
                        """
                        # Ensure the input is a 3D tensor
                        assert pretrained_pos_embed.ndim == 3, f"Expected 3D tensor, got {pretrained_pos_embed.ndim}D"
                        
                        # Get dimensions
                        _, num_patches_old, embed_dim = pretrained_pos_embed.shape
                        _, num_patches_new, embed_dim_new = target_shape
                        
                        assert embed_dim == embed_dim_new, f"Embedding dimensions must match: {embed_dim} vs {embed_dim_new}"
                        
                        # Reshape to [1, sqrt(num_patches_old), sqrt(num_patches_old), embed_dim] for interpolation
                        grid_size_old = int(num_patches_old ** 0.5)  # Assuming square grid (e.g., for ViT)
                        grid_size_new = int(num_patches_new ** 0.5)  # Target grid size
                        
                        # Reshape for interpolation
                        pos_embed = pretrained_pos_embed.reshape([1, grid_size_old, grid_size_old, embed_dim])
                        
                        # Interpolate to new grid size
                        pos_embed = pos_embed.transpose([0, 3, 1, 2])  # [1, embed_dim, grid_size_old, grid_size_old]
                        pos_embed = F.interpolate(pos_embed, size=(grid_size_new, grid_size_new), mode='bicubic', align_corners=False)
                        pos_embed = pos_embed.transpose([0, 2, 3, 1])  # [1, grid_size_new, grid_size_new, embed_dim]
                        
                        # Flatten back to [1, num_patches_new, embed_dim]
                        pos_embed = pos_embed.reshape([1, num_patches_new, embed_dim])
                        
                        return pos_embed
                    new_state_dict[k1_mapped] = resize_pos_embed(
                        pretrained_pos_embed=state_dict[k1_mapped].clone(),
                        target_shape=state_dict[k1_mapped].shape
                    )
                else:
                    overlap_dim = min(params[k1].shape[-1], state_dict[k1_mapped].shape[-1])
                    if not top_k:
                        top_k = overlap_dim
                    new_state_dict[k1_mapped] = state_dict[k1_mapped].clone()
                    if params[k1].ndim > 1:
                        new_state_dict[k1_mapped][:, :top_k] = params[k1][:, :top_k]
                    else:
                        new_state_dict[k1_mapped][:top_k] = params[k1][:top_k]
            else:
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params {} {} !".format(
                        k1, state_dict[k1].shape, k1, params[k1].shape
                    )
                )

    model.set_state_dict(new_state_dict)
    if is_float16:
        logger.info(
            "The parameter type is float16, which is converted to float32 when loading"
        )
    logger.info("load pretrain successful from {}".format(path))
    return is_float16


def save_model(
    model,
    optimizer,
    model_path,
    logger,
    config,
    is_best=False,
    prefix="ppocr",
    **kwargs,
):
    """
    save model to the target path
    """
    _mkdir_if_not_exist(model_path, logger)
    model_prefix = os.path.join(model_path, prefix)

    if prefix == "best_accuracy":
        best_model_path = os.path.join(model_path, "best_model")
        _mkdir_if_not_exist(best_model_path, logger)

    paddle.save(optimizer.state_dict(), model_prefix + ".pdopt")
    if prefix == "best_accuracy":
        paddle.save(
            optimizer.state_dict(), os.path.join(best_model_path, "model.pdopt")
        )

    is_nlp_model = config["Architecture"]["model_type"] == "kie" and config[
        "Architecture"
    ]["algorithm"] not in ["SDMGR"]
    if is_nlp_model is not True:
        paddle.save(model.state_dict(), model_prefix + ".pdparams")
        metric_prefix = model_prefix

        if prefix == "best_accuracy":
            paddle.save(
                model.state_dict(), os.path.join(best_model_path, "model.pdparams")
            )

    else:  # for kie system, we follow the save/load rules in NLP
        if config["Global"]["distributed"]:
            arch = model._layers
        else:
            arch = model
        if config["Architecture"]["algorithm"] in ["Distillation"]:
            arch = arch.Student
        arch.backbone.model.save_pretrained(model_prefix)
        metric_prefix = os.path.join(model_prefix, "metric")

        if prefix == "best_accuracy":
            arch.backbone.model.save_pretrained(best_model_path)

    save_model_info = kwargs.pop("save_model_info", False)
    if save_model_info:
        with open(os.path.join(model_path, f"{prefix}.info.json"), "w") as f:
            json.dump(kwargs, f)
        logger.info("Already save model info in {}".format(model_path))
        if prefix != "latest":
            done_flag = kwargs.pop("done_flag", False)
            update_train_results(config, prefix, save_model_info, done_flag=done_flag)

    # save metric and config
    with open(metric_prefix + ".states", "wb") as f:
        pickle.dump(kwargs, f, protocol=2)
    if is_best:
        logger.info("save best model is to {}".format(model_prefix))
    else:
        logger.info("save model in {}".format(model_prefix))

    push_to_hub = kwargs.pop("push_to_hub", False)
    if push_to_hub:
        try:
            repo_id = kwargs.pop("repo_id", None)
            repo_type = kwargs.pop("repo_type", None)
            ignore_patterns = kwargs.pop("ignore_patterns", None)
            run_as_future = kwargs.pop("run_as_future", None)
            token = kwargs.pop("hf_token", None)
            commit_message = kwargs.pop(
                "commit_message",
                f"Upload {prefix} model" + (" (best model)" if is_best else ""),
            )
    
            if repo_id is None:
                logger.warning("repo_id not provided, cannot push to Hugging Face Hub")
                return
    
            # Initialize HF API
            api = HfApi(token=token)
    
            # Push to Hub
            logger.info(f"Pushing model to Hugging Face Hub: {repo_id}")
            api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                repo_type=repo_type,
                ignore_patterns=ignore_patterns,
                run_as_future=run_as_future,
                commit_message=commit_message,
            )
    
            logger.info(f"Model successfully pushed to HF Hub: {repo_id}")
        except ImportError:
            logger.warning("huggingface_hub not installed. Cannot push to Hub.")
        except Exception as e:
            logger.error(f"Failed to push to Hugging Face Hub: {str(e)}")


def update_train_results(config, prefix, metric_info, done_flag=False, last_num=5):
    if paddle.distributed.get_rank() != 0:
        return

    assert last_num >= 1
    train_results_path = os.path.join(
        config["Global"]["save_model_dir"], "train_result.json"
    )
    save_model_tag = ["pdparams", "pdopt", "pdstates"]
    paddle_version = version.parse(paddle.__version__)
    if FLAGS_json_format_model or paddle_version >= version.parse("3.0.0"):
        save_inference_files = {
            "inference_config": "inference.yml",
            "pdmodel": "inference.json",
            "pdiparams": "inference.pdiparams",
        }
    else:
        save_inference_files = {
            "inference_config": "inference.yml",
            "pdmodel": "inference.pdmodel",
            "pdiparams": "inference.pdiparams",
            "pdiparams.info": "inference.pdiparams.info",
        }
    if os.path.exists(train_results_path):
        with open(train_results_path, "r") as fp:
            train_results = json.load(fp)
    else:
        train_results = {}
        train_results["model_name"] = config["Global"]["model_name"]
        label_dict_path = config["Global"].get("character_dict_path", "")
        if label_dict_path != "":
            label_dict_path = os.path.abspath(label_dict_path)
            if not os.path.exists(label_dict_path):
                label_dict_path = ""
        train_results["label_dict"] = label_dict_path
        train_results["train_log"] = "train.log"
        train_results["visualdl_log"] = ""
        train_results["config"] = "config.yaml"
        train_results["models"] = {}
        for i in range(1, last_num + 1):
            train_results["models"][f"last_{i}"] = {}
        train_results["models"]["best"] = {}
    train_results["done_flag"] = done_flag
    if "best" in prefix:
        if "acc" in metric_info["metric"]:
            metric_score = metric_info["metric"]["acc"]
        elif "precision" in metric_info["metric"]:
            metric_score = metric_info["metric"]["precision"]
        elif "exp_rate" in metric_info["metric"]:
            metric_score = metric_info["metric"]["exp_rate"]
        else:
            raise ValueError("No metric score found.")
        train_results["models"]["best"]["score"] = metric_score
        for tag in save_model_tag:
            if tag == "pdparams" and encrypted:
                train_results["models"]["best"][tag] = os.path.join(
                    prefix,
                    (
                        f"{prefix}.encrypted.{tag}"
                        if tag != "pdstates"
                        else f"{prefix}.states"
                    ),
                )
            else:
                train_results["models"]["best"][tag] = os.path.join(
                    prefix,
                    f"{prefix}.{tag}" if tag != "pdstates" else f"{prefix}.states",
                )
        for key in save_inference_files:
            train_results["models"]["best"][key] = os.path.join(
                prefix, "inference", save_inference_files[key]
            )
    else:
        for i in range(last_num - 1, 0, -1):
            train_results["models"][f"last_{i + 1}"] = train_results["models"][
                f"last_{i}"
            ].copy()
        if "acc" in metric_info["metric"]:
            metric_score = metric_info["metric"]["acc"]
        elif "precision" in metric_info["metric"]:
            metric_score = metric_info["metric"]["precision"]
        elif "exp_rate" in metric_info["metric"]:
            metric_score = metric_info["metric"]["exp_rate"]
        else:
            metric_score = 0
        train_results["models"][f"last_{1}"]["score"] = metric_score
        for tag in save_model_tag:
            if tag == "pdparams" and encrypted:
                train_results["models"][f"last_{1}"][tag] = os.path.join(
                    prefix,
                    (
                        f"{prefix}.encrypted.{tag}"
                        if tag != "pdstates"
                        else f"{prefix}.states"
                    ),
                )
            else:
                train_results["models"][f"last_{1}"][tag] = os.path.join(
                    prefix,
                    f"{prefix}.{tag}" if tag != "pdstates" else f"{prefix}.states",
                )
        for key in save_inference_files:
            train_results["models"][f"last_{1}"][key] = os.path.join(
                prefix, "inference", save_inference_files[key]
            )

    with open(train_results_path, "w") as fp:
        json.dump(train_results, fp)
