import torch.nn as nn
import torchvision


def get_encoders(
    model_name="efficientnet_b1",
    device="cpu",
    weights_rgb=None,
    weights_depth=None,
    do_freeze=False,
    disable_bn_running_stats=False,
    norm_layer_type=None,
    out_dim=256,
):
    if norm_layer_type == "batch":
        print(f"WARN: {norm_layer_type=}")
        norm_layer = nn.BatchNorm2d
    else:
        norm_layer = nn.Identity

    assert model_name in [
        "regnet_y_800mf",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_v2_s",
        "mobilenet_v3_small",
        "resnet18",
        "resnet50",
    ], model_name

    if model_name in ["regnet_y_800mf", "efficientnet_v2_s"]:
        print(f"WARN: {model_name} should not be used with CNNLSTM because of the batchnorm")

    if model_name == "regnet_y_800mf":
        weights = torchvision.models.RegNet_Y_800MF_Weights.IMAGENET1K_V2
        encoder_rgb = torchvision.models.regnet_y_800mf(weights=weights if weights_rgb == "imagenet" else None)
        encoder_depth = torchvision.models.regnet_y_800mf(weights=weights if weights_depth == "imagenet" else None)
        encoder_depth.stem[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for m in [encoder_rgb, encoder_depth]:
            m.fc = nn.Sequential(
                nn.Linear(784, out_dim),
            )
    elif model_name == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        encoder_rgb = torchvision.models.efficientnet_b0(
            weights=weights if weights_rgb == "imagenet" else None, norm_layer=norm_layer
        )
        encoder_depth = torchvision.models.efficientnet_b0(
            weights=weights if weights_depth == "imagenet" else None, norm_layer=norm_layer
        )
        encoder_depth.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for m in [encoder_rgb, encoder_depth]:
            m.classifier = nn.Sequential(
                nn.Linear(1280, out_dim),
            )
    elif model_name == "efficientnet_b1":
        weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V2
        encoder_rgb = torchvision.models.efficientnet_b1(
            weights=weights if weights_rgb == "imagenet" else None, norm_layer=norm_layer
        )
        encoder_depth = torchvision.models.efficientnet_b1(
            weights=weights if weights_depth == "imagenet" else None, norm_layer=norm_layer
        )
        encoder_depth.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for m in [encoder_rgb, encoder_depth]:
            m.classifier = nn.Sequential(
                nn.Linear(1280, out_dim),
            )
    elif model_name == "mobilenet_v3_small":
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        encoder_rgb = torchvision.models.mobilenet_v3_small(
            weights=weights if weights_rgb == "imagenet" else None, norm_layer=norm_layer
        )
        encoder_depth = torchvision.models.mobilenet_v3_small(
            weights=weights if weights_depth == "imagenet" else None, norm_layer=norm_layer
        )
        encoder_depth.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for m in [encoder_rgb, encoder_depth]:
            m.classifier = nn.Sequential(
                nn.Linear(576, out_dim),
            )
    elif model_name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        encoder_rgb = torchvision.models.resnet18(
            weights=weights if weights_rgb == "imagenet" else None, norm_layer=norm_layer
        )
        encoder_depth = torchvision.models.resnet18(
            weights=weights if weights_depth == "imagenet" else None, norm_layer=norm_layer
        )
        encoder_depth.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for m in [encoder_rgb, encoder_depth]:
            m.fc = nn.Sequential(
                nn.Linear(512, out_dim),
            )
    elif model_name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        encoder_rgb = torchvision.models.resnet50(
            weights=weights if weights_rgb == "imagenet" else None, norm_layer=norm_layer
        )
        encoder_depth = torchvision.models.resnet50(
            weights=weights if weights_depth == "imagenet" else None, norm_layer=norm_layer
        )
        encoder_depth.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for m in [encoder_rgb, encoder_depth]:
            m.fc = nn.Sequential(
                nn.Linear(512, out_dim),
            )
    else:
        weights = torchvision.models.EfficientNet_V2_Weights.IMAGENET1K_V1
        encoder_rgb = torchvision.models.efficientnet_v2_s(weights=weights if weights_depth == "imagenet" else None)
        encoder_depth = torchvision.models.efficientnet_v2_s(weights=weights if weights_depth == "imagenet" else None)
        encoder_depth.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for m in [encoder_rgb, encoder_depth]:
            m.classifier = nn.Sequential(
                nn.Linear(1280, out_dim),
            )
    for m in [encoder_rgb, encoder_depth]:
        m.to(device)
        if disable_bn_running_stats:
            disable_running_stats(m)
    if do_freeze:
        to_freeze = []
        if weights_rgb is not None:
            to_freeze.append(encoder_rgb)
        if weights_depth is not None:
            to_freeze.append(encoder_depth)
        for m in to_freeze:
            for name, param in m.named_parameters():
                if name.startswith("classifier") or name.startswith("fc"):
                    continue
                param.requires_grad = False
    return encoder_rgb, encoder_depth


def disable_running_stats(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None


def is_param_part_of_encoders(name):
    return "encoder_rgb" in name or "encoder_depth" in name
