def freeze_svtrnet_backbone(model, logger):
    """
    Freeze SVTRNet backbone layers for transfer learning with dimension changes
    Unfreeze patch_embed and pos_embed due to shape changes (32x640 -> 48x480)
    Keep blocks3 + final layers + neck + head trainable
    """
    backbone = model.backbone

    # UNFREEZE patch_embed due to aspect ratio change (1:20 -> 1:10)
    for param in backbone.patch_embed.parameters():
        param.stop_gradient = False

    # UNFREEZE pos_embed due to different number of patches
    backbone.pos_embed.stop_gradient = False

    # Freeze first two transformer stages
    for param in backbone.blocks1.parameters():
        param.stop_gradient = True
    for param in backbone.blocks2.parameters():
        param.stop_gradient = True

    # Freeze subsampling layers
    if hasattr(backbone, "sub_sample1"):
        for param in backbone.sub_sample1.parameters():
            param.stop_gradient = True
    if hasattr(backbone, "sub_sample2"):
        for param in backbone.sub_sample2.parameters():
            param.stop_gradient = True

    # Count and log parameters
    total_params, trainable_params = count_parameters(model)
    frozen_params = total_params - trainable_params

    # Handle both logger objects and print function
    log_func = logger.info if hasattr(logger, "info") else logger

    log_func("=" * 60)
    log_func("PARAMETER FREEZING SUMMARY")
    log_func("=" * 60)
    log_func(f"Total parameters: {total_params:,}")
    log_func(
        f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)"
    )
    log_func(
        f"Frozen parameters: {frozen_params:,} ({frozen_params / total_params * 100:.1f}%)"
    )
    log_func("")
    log_func("FROZEN LAYERS:")
    log_func("  - blocks1 (first transformer stage)")
    log_func("  - blocks2 (second transformer stage)")
    log_func("  - sub_sample1, sub_sample2 (downsampling layers)")
    log_func("")
    log_func("TRAINABLE LAYERS:")
    log_func("  - patch_embed (due to aspect ratio change: 1:20 -> 1:10)")
    log_func("  - pos_embed (due to different patch count)")
    log_func("  - blocks3 (third transformer stage)")
    log_func("  - final backbone layers (avg_pool, last_conv, etc.)")
    log_func("  - neck (SequenceEncoder)")
    log_func("  - head (CTCHead)")
    log_func("=" * 60)

    # Detailed parameter breakdown
    log_detailed_parameter_info(model, logger)


def count_parameters(model):
    """Count total and trainable parameters in the model"""
    total_params = sum(p.numel().item() for p in model.parameters())
    trainable_params = sum(
        p.numel().item() for p in model.parameters() if not p.stop_gradient
    )
    return total_params, trainable_params


def log_detailed_parameter_info(model, logger):
    """Log detailed parameter information for each component"""
    # Handle both logger objects and print function
    log_func = logger.info if hasattr(logger, "info") else logger

    log_func("DETAILED PARAMETER BREAKDOWN:")
    log_func("-" * 50)

    # Backbone components
    if hasattr(model, "backbone"):
        backbone = model.backbone

        # Patch embedding
        patch_embed_params = sum(
            p.numel().item() for p in backbone.patch_embed.parameters()
        )
        patch_embed_trainable = sum(
            p.numel().item()
            for p in backbone.patch_embed.parameters()
            if not p.stop_gradient
        )
        log_func(
            f"patch_embed: {patch_embed_params:,} total, {patch_embed_trainable:,} trainable"
        )

        # Position embedding
        pos_embed_params = backbone.pos_embed.numel().item()
        pos_embed_trainable = (
            0 if backbone.pos_embed.stop_gradient else pos_embed_params
        )
        log_func(
            f"pos_embed: {pos_embed_params:,} total, {pos_embed_trainable:,} trainable"
        )

        # Transformer blocks
        for i, block_layer in enumerate(
            [backbone.blocks1, backbone.blocks2, backbone.blocks3], 1
        ):
            block_params = sum(p.numel().item() for p in block_layer.parameters())
            block_trainable = sum(
                p.numel().item()
                for p in block_layer.parameters()
                if not p.stop_gradient
            )
            log_func(
                f"blocks{i}: {block_params:,} total, {block_trainable:,} trainable"
            )

        # Subsampling layers
        if hasattr(backbone, "sub_sample1"):
            sub1_params = sum(
                p.numel().item() for p in backbone.sub_sample1.parameters()
            )
            sub1_trainable = sum(
                p.numel().item()
                for p in backbone.sub_sample1.parameters()
                if not p.stop_gradient
            )
            log_func(
                f"sub_sample1: {sub1_params:,} total, {sub1_trainable:,} trainable"
            )

        if hasattr(backbone, "sub_sample2"):
            sub2_params = sum(
                p.numel().item() for p in backbone.sub_sample2.parameters()
            )
            sub2_trainable = sum(
                p.numel().item()
                for p in backbone.sub_sample2.parameters()
                if not p.stop_gradient
            )
            log_func(
                f"sub_sample2: {sub2_params:,} total, {sub2_trainable:,} trainable"
            )

        # Final backbone layers
        final_layers = ["avg_pool", "last_conv", "hardswish", "dropout"]
        for layer_name in final_layers:
            if hasattr(backbone, layer_name):
                layer = getattr(backbone, layer_name)
                if hasattr(layer, "parameters"):
                    layer_params = sum(p.numel().item() for p in layer.parameters())
                    layer_trainable = sum(
                        p.numel().item()
                        for p in layer.parameters()
                        if not p.stop_gradient
                    )
                    log_func(
                        f"{layer_name}: {layer_params:,} total, {layer_trainable:,} trainable"
                    )

    # Neck
    if hasattr(model, "neck") and model.neck is not None:
        neck_params = sum(p.numel().item() for p in model.neck.parameters())
        neck_trainable = sum(
            p.numel().item() for p in model.neck.parameters() if not p.stop_gradient
        )
        log_func(f"neck: {neck_params:,} total, {neck_trainable:,} trainable")

    # Head
    if hasattr(model, "head"):
        head_params = sum(p.numel().item() for p in model.head.parameters())
        head_trainable = sum(
            p.numel().item() for p in model.head.parameters() if not p.stop_gradient
        )
        log_func(f"head: {head_params:,} total, {head_trainable:,} trainable")

    log_func("-" * 50)


def verify_freeze_status(model, logger):
    """Verify that parameters are correctly frozen/unfrozen"""
    # Handle both logger objects and print function
    log_func = logger.info if hasattr(logger, "info") else logger

    log_func("VERIFYING FREEZE STATUS:")
    log_func("-" * 40)

    if hasattr(model, "backbone"):
        backbone = model.backbone

        # Check patch_embed (should be trainable)
        patch_embed_frozen = all(
            p.stop_gradient for p in backbone.patch_embed.parameters()
        )
        log_func(f"patch_embed frozen: {patch_embed_frozen} (should be False)")

        # Check pos_embed (should be trainable)
        pos_embed_frozen = backbone.pos_embed.stop_gradient
        log_func(f"pos_embed frozen: {pos_embed_frozen} (should be False)")

        # Check blocks1 (should be frozen)
        blocks1_frozen = all(p.stop_gradient for p in backbone.blocks1.parameters())
        log_func(f"blocks1 frozen: {blocks1_frozen} (should be True)")

        # Check blocks2 (should be frozen)
        blocks2_frozen = all(p.stop_gradient for p in backbone.blocks2.parameters())
        log_func(f"blocks2 frozen: {blocks2_frozen} (should be True)")

        # Check blocks3 (should be trainable)
        blocks3_frozen = all(p.stop_gradient for p in backbone.blocks3.parameters())
        log_func(f"blocks3 frozen: {blocks3_frozen} (should be False)")

        # Verify subsampling layers
        if hasattr(backbone, "sub_sample1"):
            sub1_frozen = all(
                p.stop_gradient for p in backbone.sub_sample1.parameters()
            )
            log_func(f"sub_sample1 frozen: {sub1_frozen} (should be True)")

        if hasattr(backbone, "sub_sample2"):
            sub2_frozen = all(
                p.stop_gradient for p in backbone.sub_sample2.parameters()
            )
            log_func(f"sub_sample2 frozen: {sub2_frozen} (should be True)")

    log_func("-" * 40)