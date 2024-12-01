import copy

import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models import resnet50


def get_hook(outs, name):
    def hook(self, input, output):
        outs[name] = output

    return hook


class DETR(nn.Module):

    def __init__(self, num_classes, d_model=256, n_tokens=15 * 20, n_layers=6, n_heads=8, n_queries=100):
        super().__init__()

        self.num_classes = num_classes
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_queries = n_queries

        self.backbone = resnet50()

        self.conv1x1 = nn.Conv2d(2048, d_model, kernel_size=1, stride=1)

        self.pe_encoder = nn.Parameter(torch.rand((1, n_tokens, d_model)), requires_grad=True)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, dropout=0.1
        )

        self.t_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.queries = nn.Parameter(torch.rand((1, n_queries, d_model)), requires_grad=True)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True, dropout=0.1
        )

        self.t_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.class_mlps = get_clones(nn.Linear(d_model, num_classes), n_layers)
        self.bbox_mlps = get_clones(nn.Linear(d_model, 4), n_layers)

        # Add hooks to get intermediate outcomes
        self.decoder_outs = {}
        for i, layer in enumerate(self.t_decoder.layers):
            name = f"layer_{i}"
            layer.register_forward_hook(get_hook(self.decoder_outs, name))

        self.backbone_feats = {}

        def hook_resnet50_feats(model, inp, out):
            self.backbone_feats["layer4"] = out

        self.backbone.layer4.register_forward_hook(hook_resnet50_feats)

    def forward(self, x):
        _ = self.backbone(x)
        tokens = self.backbone_feats["layer4"]
        tokens = self.conv1x1(tokens)
        tokens = rearrange(tokens, "b c h w -> b (h w) c")

        out_encoder = self.t_encoder(tokens + self.pe_encoder)

        out_decoder = self.t_decoder(self.queries.repeat(len(out_encoder), 1, 1), out_encoder)

        outs = []
        for layer_idx, (n, o) in enumerate(sorted(self.decoder_outs.items())):
            pred_logits = self.class_mlps[layer_idx](o)
            pred_boxes = self.bbox_mlps[layer_idx](o)
            pred_boxes = torch.sigmoid(pred_boxes)
            outs.append({"pred_logits": pred_logits, "pred_boxes": pred_boxes})
        last_out = outs.pop()
        return {
            "pred_logits": last_out["pred_logits"],
            "pred_boxes": last_out["pred_boxes"],
            "aux_outputs": outs,
        }


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
