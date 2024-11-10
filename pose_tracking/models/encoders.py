import torch
import torch.nn as nn
import torchvision


def get_encoders(model_name="regnet_y_800mf", device="cpu"):
    assert model_name in ["regnet_y_800mf", "efficientnet_v2_s"], model_name
    if model_name == "regnet_y_800mf":
        encoder_s_img = torchvision.models.regnet_y_800mf(
            weights=torchvision.models.RegNet_Y_800MF_Weights.IMAGENET1K_V2
        )
        encoder_s_depth = torchvision.models.regnet_y_800mf(
            weights=torchvision.models.RegNet_Y_800MF_Weights.IMAGENET1K_V2
        )
        encoder_s_depth.stem[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for m in [encoder_s_img, encoder_s_depth]:
            m.fc = nn.Sequential(
                nn.Linear(784, 256),
            )
    else:
        encoder_s_img = torchvision.models.efficientnet_v2_s(
            weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        encoder_s_depth = torchvision.models.efficientnet_v2_s(
            weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
        )
        encoder_s_depth.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for m in [encoder_s_img, encoder_s_depth]:
            m.classifier = nn.Sequential(
                nn.Linear(1280, 256),
            )
    for m in [encoder_s_img, encoder_s_depth]:
        m.to(device)
    return encoder_s_img, encoder_s_depth
