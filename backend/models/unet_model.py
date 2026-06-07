import segmentation_models_pytorch as smp


def create_unet():

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=5
    )

    return model