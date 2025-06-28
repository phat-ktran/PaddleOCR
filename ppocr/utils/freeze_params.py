def freeze_svtrnet_backbone(model, logger):
    """
    Freeze SVTRNet backbone layers for transfer learning
    Keep blocks3 + final layers + neck + head trainable
    """
    backbone = model.backbone
    
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

    logger.info("Frozen: blocks1, blocks2, sub_sample layers")
    logger.info("Trainable: pos_embed, blocks3, final layers, neck, head")

