import torch
import torch.nn as nn
import torchvision


def get_encoders(
    model_name="efficientnet_b1",
    device="cpu",
    weights_rgb="imagenet",
    weights_depth="imagenet",
    do_freeze=False,
    disable_bn_running_stats=False,
    norm_layer_type=None,
    out_dim=256,
    ignored_modalities="",
    dropout=0.0,
):
    if norm_layer_type == "bn":
        print(f"WARN: {norm_layer_type=}")
        norm_layer = nn.BatchNorm2d
    elif norm_layer_type == "frozen_bn":
        norm_layer = FrozenBatchNorm2d
    else:
        norm_layer = nn.Identity

    if dropout > 0:
        dropout_layer = nn.Dropout(dropout)
    else:
        dropout_layer = nn.Identity()

    assert model_name in [
        "regnet_y_800mf",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_v2_s",
        "mobilenet_v3_small",
        "resnet18",
        "resnet50",
    ], model_name

    if model_name == "regnet_y_800mf":
        weights = torchvision.models.RegNet_Y_800MF_Weights.IMAGENET1K_V2
        encoder_rgb = torchvision.models.regnet_y_800mf(weights=weights if weights_rgb == "imagenet" else None)
        encoder_depth = torchvision.models.regnet_y_800mf(weights=weights if weights_depth == "imagenet" else None)
        encoder_depth.stem[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for m in [encoder_rgb, encoder_depth]:
            m.fc = nn.Sequential(
                dropout_layer,
                nn.Linear(784, out_dim),
            )
    elif model_name == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        encoder_rgb = torchvision.models.efficientnet_b0(weights=weights if weights_rgb == "imagenet" else None)
        encoder_depth = torchvision.models.efficientnet_b0(weights=weights if weights_depth == "imagenet" else None)
        encoder_depth.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for m in [encoder_rgb, encoder_depth]:
            m.classifier = nn.Sequential(
                dropout_layer,
                nn.Linear(1280, out_dim),
            )
    elif model_name == "efficientnet_b1":
        weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V2
        encoder_rgb = torchvision.models.efficientnet_b1(weights=weights if weights_rgb == "imagenet" else None)
        encoder_depth = torchvision.models.efficientnet_b1(weights=weights if weights_depth == "imagenet" else None)
        encoder_depth.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for m in [encoder_rgb, encoder_depth]:
            m.classifier = nn.Sequential(
                dropout_layer,
                nn.Linear(1280, out_dim),
            )
    elif model_name == "mobilenet_v3_small":
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        encoder_rgb = torchvision.models.mobilenet_v3_small(weights=weights if weights_rgb == "imagenet" else None)
        encoder_depth = torchvision.models.mobilenet_v3_small(weights=weights if weights_depth == "imagenet" else None)
        encoder_depth.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for m in [encoder_rgb, encoder_depth]:
            m.classifier = nn.Sequential(
                dropout_layer,
                nn.Linear(576, out_dim),
            )
    elif model_name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        encoder_rgb = torchvision.models.resnet18(weights=weights if weights_rgb == "imagenet" else None)
        encoder_depth = torchvision.models.resnet18(weights=weights if weights_depth == "imagenet" else None)
        encoder_depth.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=encoder_rgb.conv1.kernel_size,
            stride=encoder_rgb.conv1.stride,
            padding=encoder_rgb.conv1.padding,
            bias=False,
        )
        for m in [encoder_rgb, encoder_depth]:
            m.fc = nn.Sequential(
                dropout_layer,
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
        encoder_depth.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=encoder_rgb.conv1.kernel_size,
            stride=encoder_rgb.conv1.stride,
            padding=encoder_rgb.conv1.padding,
            bias=False,
        )
        for m in [encoder_rgb, encoder_depth]:
            m.fc = nn.Sequential(
                dropout_layer,
                nn.Linear(2048, out_dim),
            )
    elif model_name == "dino":
        encoder_rgb = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
        for p in encoder_rgb.parameters():
            p.requires_grad = False
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        encoder_depth = torchvision.models.resnet50(
            weights=weights if weights_depth == "imagenet" else None, norm_layer=norm_layer
        )
        encoder_depth.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=encoder_depth.conv1.kernel_size,
            stride=encoder_depth.conv1.stride,
            padding=encoder_depth.conv1.padding,
            bias=False,
        )
        for m in [encoder_depth]:
            m.fc = nn.Sequential(
                dropout_layer,
                nn.Linear(2048, out_dim),
            )
    else:
        weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        encoder_rgb = torchvision.models.efficientnet_v2_s(weights=weights if weights_depth == "imagenet" else None)
        encoder_depth = torchvision.models.efficientnet_v2_s(weights=weights if weights_depth == "imagenet" else None)
        encoder_depth.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        for m in [encoder_rgb, encoder_depth]:
            m.classifier = nn.Sequential(
                dropout_layer,
                nn.Linear(1280, out_dim),
            )
    for m in [encoder_rgb, encoder_depth]:
        m.to(device)
        if disable_bn_running_stats:
            disable_running_stats(m)
        if norm_layer_type != "bn":
            m = replace_batchnorm(m, norm_layer)
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
    if ignored_modalities:

        def tmp(x, *args, **kwargs):
            return torch.zeros((x.shape[0], out_dim), device=x.device)

        if "depth" in ignored_modalities:
            encoder_depth = tmp
        if "rgb" in ignored_modalities:
            encoder_rgb = tmp
    return encoder_rgb, encoder_depth


def disable_running_stats(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None


def is_param_part_of_encoders(name, encoder_module_prefix="encoder"):
    return encoder_module_prefix in name


def replace_batchnorm(model, norm_layer):
    """
    Recursively replace all nn.BatchNorm2d layers in a model with some other layer.

    Args:
        model (nn.Module): The model to modify.

    Returns:
        nn.Module: The modified model with nn.BatchNorm2d replaced.
    """
    for name, module in model.named_children():
        # Recursively replace in child modules
        if isinstance(module, nn.BatchNorm2d):
            # Replace nn.BatchNorm2d with nn.FrozenBatchNorm2d
            setattr(model, name, norm_layer(module.num_features))
        else:
            replace_batchnorm(module, norm_layer)  # Recurse for submodules
    return model


class FrozenBatchNorm2d(torch.nn.Module):
    """
    # https://github.com/fundamentalvision/Deformable-DETR
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
