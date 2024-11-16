import torch.nn as nn
import torchvision


def get_encoders(model_name="regnet_y_800mf", device="cpu", weights_img="imagenet", weights_depth=None):
    assert model_name in ["regnet_y_800mf", "efficientnet_b1", "efficientnet_v2_s"], model_name
    if model_name == "regnet_y_800mf":
        weights = torchvision.models.RegNet_Y_800MF_Weights.IMAGENET1K_V2
        encoder_s_img = torchvision.models.regnet_y_800mf(weights=weights if weights_img == "imagenet" else None)
        encoder_s_depth = torchvision.models.regnet_y_800mf(weights=weights if weights_depth == "imagenet" else None)
        encoder_s_depth.stem[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for m in [encoder_s_img, encoder_s_depth]:
            m.fc = nn.Sequential(
                nn.Linear(784, 256),
            )
    elif model_name == "efficientnet_b1":
        weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V2
        encoder_s_img = torchvision.models.efficientnet_b1(weights=weights if weights_img == "imagenet" else None)
        encoder_s_depth = torchvision.models.efficientnet_b1(weights=weights if weights_depth == "imagenet" else None)
        encoder_s_depth.stem[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for m in [encoder_s_img, encoder_s_depth]:
            m.classifier = nn.Sequential(
                nn.Linear(1280, 256),
            )
    else:
        weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        encoder_s_img = torchvision.models.efficientnet_v2_s(weights=weights if weights_depth == "imagenet" else None)
        encoder_s_depth = torchvision.models.efficientnet_v2_s(weights=weights if weights_depth == "imagenet" else None)
        encoder_s_depth.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for m in [encoder_s_img, encoder_s_depth]:
            m.classifier = nn.Sequential(
                nn.Linear(1280, 256),
            )
    for m in [encoder_s_img, encoder_s_depth]:
        m.to(device)
    return encoder_s_img, encoder_s_depth


def is_param_part_of_encoders(name):
    return "encoder_img" in name or "encoder_depth" in name
