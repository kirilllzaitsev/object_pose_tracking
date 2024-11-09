import torch
import torch.nn as nn
from torchvision import models


class DirectRegrCNN(nn.Module):
    def __init__(self, dropout_prob=0.3, backbone_name="mobilenet_v3_large"):
        super().__init__()

        if backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=True)
            last_backbone_ch = 2048
        elif backbone_name == "mobilenet_v3_large":
            backbone = models.mobilenet_v3_large(pretrained=True)
            last_backbone_ch = 960
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        self.backbone_name = backbone_name

        self.do_modify_first_conv = True
        self.do_modify_first_conv = False
        if self.do_modify_first_conv:
            self.modify_first_conv(backbone)

        modules = list(backbone.children())[:-2]
        self.features = nn.Sequential(*modules)

        self.translation_head = nn.Sequential(
            nn.Conv2d(last_backbone_ch, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 3),
        )

        self.rotation_head = nn.Sequential(
            nn.Conv2d(last_backbone_ch, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 4),
            nn.Tanh(),
        )

    def modify_first_conv(self, backbone):
        if self.backbone_name == "mobilenet_v3_large":
            original_conv = backbone.features[0][0]
        else:
            original_conv = backbone.conv1

        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = original_conv.weight
            new_conv.weight[:, 3:, :, :] = original_conv.weight.mean(dim=1, keepdim=True)

        if self.backbone_name == "mobilenet_v3_large":
            backbone.features[0][0] = new_conv
        else:
            backbone.conv1 = new_conv

    def forward(self, x, segm_mask):
        if self.do_modify_first_conv:
            x = torch.cat([x, segm_mask], dim=1)
        features = self.features(x)

        trans_output = self.translation_head(features)

        rot_output = self.rotation_head(features)

        rot_output = rot_output / rot_output.norm(dim=1, keepdim=True)

        return trans_output, rot_output
