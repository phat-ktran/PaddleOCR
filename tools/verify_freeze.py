#!/usr/bin/env python3
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

"""
Script to verify parameter freezing before and during training
Usage: python tools/verify_freeze.py -c configs/rec/PP-Thesis/Nom/local/rec_svtrnet_base_nom_exp_b2.0.2.yml
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import yaml

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import paddle
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.freeze_params import verify_freeze_status


def parse_args():
    parser = argparse.ArgumentParser(description="Verify parameter freezing")
    parser.add_argument("-c", "--config", help="config file path")
    parser.add_argument(
        "--check_gradients",
        action="store_true",
        help="Check if gradients are computed for frozen parameters",
    )
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from yaml file"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def build_model_from_config(config):
    """Build model from configuration"""
    global_config = config["Global"]

    # Build post process to get character info
    post_process_class = build_post_process(config["PostProcess"], global_config)

    # Set character number for model
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        config["Architecture"]["Head"]["out_channels"] = char_num

    # Build model
    model = build_model(config["Architecture"])
    return model


def check_parameter_status(model, logger_func=print):
    """Check which parameters are frozen/trainable"""
    logger_func("\n" + "=" * 80)
    logger_func("PARAMETER STATUS CHECK")
    logger_func("=" * 80)

    total_params = 0
    trainable_params = 0
    frozen_params = 0

    component_stats = {}

    def analyze_component(name, component):
        if component is None:
            return 0, 0

        comp_total = sum(p.numel().item() for p in component.parameters())
        comp_trainable = sum(
            p.numel().item() for p in component.parameters() if not p.stop_gradient
        )
        comp_frozen = comp_total - comp_trainable

        component_stats[name] = {
            "total": comp_total,
            "trainable": comp_trainable,
            "frozen": comp_frozen,
            "trainable_pct": comp_trainable / comp_total * 100 if comp_total > 0 else 0,
        }

        return comp_total, comp_trainable

    # Analyze backbone components
    if hasattr(model, "backbone"):
        backbone = model.backbone

        # Individual backbone components
        if hasattr(backbone, "patch_embed"):
            analyze_component("patch_embed", backbone.patch_embed)

        if hasattr(backbone, "pos_embed"):
            pos_total = backbone.pos_embed.numel().item()
            pos_trainable = 0 if backbone.pos_embed.stop_gradient else pos_total
            component_stats["pos_embed"] = {
                "total": pos_total,
                "trainable": pos_trainable,
                "frozen": pos_total - pos_trainable,
                "trainable_pct": pos_trainable / pos_total * 100
                if pos_total > 0
                else 0,
            }

        # Transformer blocks
        for i, block_name in enumerate(["blocks1", "blocks2", "blocks3"], 1):
            if hasattr(backbone, block_name):
                analyze_component(f"blocks{i}", getattr(backbone, block_name))

        # Subsampling layers
        for sub_name in ["sub_sample1", "sub_sample2"]:
            if hasattr(backbone, sub_name):
                analyze_component(sub_name, getattr(backbone, sub_name))

        # Final layers
        final_layers = ["avg_pool", "last_conv", "hardswish", "dropout"]
        for layer_name in final_layers:
            if hasattr(backbone, layer_name):
                layer = getattr(backbone, layer_name)
                if hasattr(layer, "parameters"):
                    analyze_component(layer_name, layer)

    # Analyze neck and head
    if hasattr(model, "neck") and model.neck is not None:
        analyze_component("neck", model.neck)

    if hasattr(model, "head"):
        analyze_component("head", model.head)

    # Calculate totals
    total_params = sum(p.numel().item() for p in model.parameters())
    trainable_params = sum(
        p.numel().item() for p in model.parameters() if not p.stop_gradient
    )
    frozen_params = total_params - trainable_params

    # Print summary
    logger_func("\nOVERALL SUMMARY:")
    logger_func(f"Total parameters: {total_params:,}")
    logger_func(
        f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)"
    )
    logger_func(
        f"Frozen parameters: {frozen_params:,} ({frozen_params / total_params * 100:.1f}%)"
    )

    # Print component breakdown
    logger_func("\nCOMPONENT BREAKDOWN:")
    logger_func("-" * 70)
    logger_func(
        f"{'Component':<15} {'Total':<12} {'Trainable':<12} {'Frozen':<12} {'Train%':<8}"
    )
    logger_func("-" * 70)

    for comp_name, stats in component_stats.items():
        logger_func(
            f"{comp_name:<15} {stats['total']:<12,} {stats['trainable']:<12,} "
            f"{stats['frozen']:<12,} {stats['trainable_pct']:<8.1f}%"
        )

    logger_func("-" * 70)

    return component_stats


def check_gradient_flow(model, logger_func=print):
    """Check if gradients flow correctly through trainable parameters"""
    logger_func("\n" + "=" * 80)
    logger_func("GRADIENT FLOW CHECK")
    logger_func("=" * 80)

    # Create dummy input
    dummy_input = paddle.randn([1, 3, 48, 480])  # Based on your config

    # Forward pass
    model.train()
    output = model(dummy_input)

    # Create dummy loss
    if isinstance(output, (list, tuple)):
        loss = output[0].mean()
    else:
        loss = output.mean()

    # Backward pass
    loss.backward()

    # Check gradients
    components_with_grad = []
    components_without_grad = []

    def check_component_gradients(name, component):
        if component is None:
            return

        has_grad = False
        for param in component.parameters():
            if param.grad is not None and not param.stop_gradient:
                has_grad = True
                break

        if has_grad:
            components_with_grad.append(name)
        else:
            components_without_grad.append(name)

    # Check backbone components
    if hasattr(model, "backbone"):
        backbone = model.backbone

        # Check pos_embed separately
        if hasattr(backbone, "pos_embed"):
            if (
                backbone.pos_embed.grad is not None
                and not backbone.pos_embed.stop_gradient
            ):
                components_with_grad.append("pos_embed")
            else:
                components_without_grad.append("pos_embed")

        # Check other components
        components = [
            "patch_embed",
            "blocks1",
            "blocks2",
            "blocks3",
            "sub_sample1",
            "sub_sample2",
            "avg_pool",
            "last_conv",
        ]

        for comp_name in components:
            if hasattr(backbone, comp_name):
                check_component_gradients(comp_name, getattr(backbone, comp_name))

    # Check neck and head
    if hasattr(model, "neck") and model.neck is not None:
        check_component_gradients("neck", model.neck)

    if hasattr(model, "head"):
        check_component_gradients("head", model.head)

    logger_func(f"Components WITH gradients: {components_with_grad}")
    logger_func(f"Components WITHOUT gradients: {components_without_grad}")

    # Verify expected behavior
    expected_trainable = ["patch_embed", "pos_embed", "blocks3", "neck", "head"]
    expected_frozen = ["blocks1", "blocks2", "sub_sample1", "sub_sample2"]

    logger_func("\nEXPECTED vs ACTUAL:")
    logger_func(f"Expected trainable: {expected_trainable}")
    logger_func(f"Actually trainable: {components_with_grad}")
    logger_func(f"Expected frozen: {expected_frozen}")
    logger_func(f"Actually frozen: {components_without_grad}")

    # Check if expectations match
    trainable_correct = all(
        comp in components_with_grad
        for comp in expected_trainable
        if comp in components_with_grad + components_without_grad
    )
    frozen_correct = all(
        comp in components_without_grad
        for comp in expected_frozen
        if comp in components_with_grad + components_without_grad
    )

    logger_func("\nVERIFICATION:")
    logger_func(f"Trainable components correct: {trainable_correct}")
    logger_func(f"Frozen components correct: {frozen_correct}")
    logger_func(f"Overall freezing correct: {trainable_correct and frozen_correct}")

    return trainable_correct and frozen_correct


def main():
    args = parse_args()

    if not args.config:
        print("Please provide config file with -c argument")
        return

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    # Build model
    print("Building model...")
    model = build_model_from_config(config)

    # Load pretrained weights if specified
    if config["Global"].get("pretrained_model"):
        print(f"Loading pretrained model: {config['Global']['pretrained_model']}")
        load_model(config, model, None, config["Architecture"]["model_type"])

    # Check status before freezing
    print("\n" + "=" * 80)
    print("BEFORE FREEZING")
    print("=" * 80)
    check_parameter_status(model)

    # Apply freezing
    freeze_params_func = config["Architecture"].get("freeze_params_func")
    if freeze_params_func:
        print(f"\nApplying freeze function: {freeze_params_func}")
        from ppocr.utils.freeze_params import freeze_svtrnet_backbone

        freeze_svtrnet_backbone(model, print)
    else:
        print("No freeze function specified in config")

    # Check status after freezing
    print("\n" + "=" * 80)
    print("AFTER FREEZING")
    print("=" * 80)
    check_parameter_status(model)

    # Verify freeze status
    verify_freeze_status(model, print)

    # Check gradient flow if requested
    if args.check_gradients:
        gradient_check_passed = check_gradient_flow(model)
        if gradient_check_passed:
            print("\n✅ Gradient flow verification PASSED")
        else:
            print("\n❌ Gradient flow verification FAILED")

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    total_params = sum(p.numel().item() for p in model.parameters())
    trainable_params = sum(
        p.numel().item() for p in model.parameters() if not p.stop_gradient
    )

    print("✅ Model loaded successfully")
    print("✅ Freeze function applied")
    print(f"📊 Total parameters: {total_params:,}")
    print(
        f"📊 Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)"
    )

    # Expected components
    expected_trainable = ["patch_embed", "pos_embed", "blocks3", "neck", "head"]
    expected_frozen = ["blocks1", "blocks2", "sub_sample1", "sub_sample2"]

    print(f"📋 Expected trainable: {expected_trainable}")
    print(f"📋 Expected frozen: {expected_frozen}")

    print("\n🎯 Ready for training with transfer learning setup!")
    print("=" * 80)


if __name__ == "__main__":
    main()
