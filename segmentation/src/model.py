import segmentation_models_pytorch as smp

def initialize_model(
    encoder_name="tu-xception41",
    atrous_rates=(6, 12, 18),
    classes=3,
    activation=None,
):
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        decoder_atrous_rates=atrous_rates,
        classes=classes,
        activation=activation,
    )

    return model
