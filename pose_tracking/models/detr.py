import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pose_tracking.models.pos_encoding import PosEncoding, SpatialPosEncoding
from pose_tracking.utils.geom import (
    backproj_2d_to_3d,
    backproj_2d_to_3d_batch,
    calibrate_2d_pts_batch,
)
from pose_tracking.utils.kpt_utils import load_extractor
from torchvision.models import resnet50


def get_hook(outs, name):
    def hook(self, input, output):
        outs[name] = output

    return hook


class DETR(nn.Module):

    def __init__(
        self,
        num_classes,
        d_model=256,
        n_tokens=15 * 20,
        n_layers=6,
        n_heads=8,
        n_queries=100,
        head_hidden_dim=256,
        head_num_layers=3,
        backbone_name="resnet18",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_queries = n_queries

        if backbone_name == "resnet50":
            self.final_layer_name = "layer4"
            self.final_feature_dim = 2048
            self.backbone_cls = resnet50
        elif backbone_name == "resnet18":
            self.final_feature_dim = 512
            self.final_layer_name = "layer4"
            self.backbone_cls = resnet18
        else:
            raise ValueError(f"Unknown backbone {backbone_name}")
        self.backbone = self.backbone_cls(norm_layer=FrozenBatchNorm2d)
        self.conv1x1 = nn.Conv2d(self.final_feature_dim, d_model, kernel_size=1, stride=1)

        self.backbone.fc = nn.Identity()

        self.pe_encoder = nn.Parameter(torch.rand((1, n_tokens, d_model)), requires_grad=True)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, dropout=0.0
        )

        self.t_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.queries = nn.Parameter(torch.rand((1, n_queries, d_model)), requires_grad=True)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True, dropout=0.0
        )

        self.t_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.class_mlps = get_clones(nn.Linear(d_model, self.num_classes), n_layers)
        self.bbox_mlps = get_clones(
            MLP(in_dim=d_model, out_dim=4, hidden_dim=head_hidden_dim, num_layers=head_num_layers),
            n_layers,
        )
        # self.t_mlps = get_clones(nn.Linear(d_model, 3), n_layers)
        # self.rot_mlps = get_clones(nn.Linear(d_model, 4), n_layers)

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


class KeypointDETR(nn.Module):

    def __init__(
        self,
        num_classes,
        kpt_extractor_name="superpoint",
        d_model=256,
        n_tokens=15 * 20,
        n_layers=6,
        n_heads=8,
        n_queries=100,
        kpt_spatial_dim=2,
        encoding_type="spatial",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_queries = n_queries

        descriptor_dim = 256
        self.conv1x1 = nn.Conv1d(descriptor_dim, d_model, kernel_size=1, stride=1)

        if encoding_type == "spatial":
            self.pe_encoder = SpatialPosEncoding(d_model, ndim=kpt_spatial_dim)
        else:
            self.pe_encoder = PosEncoding(d_model, max_len=1024)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, dropout=0.0
        )

        self.t_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.queries = nn.Parameter(torch.rand((1, n_queries, d_model)), requires_grad=True)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True, dropout=0.0
        )

        self.t_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.class_mlps = get_clones(nn.Linear(d_model, self.num_classes), n_layers)
        self.bbox_mlps = get_clones(nn.Linear(d_model, 4), n_layers)

        # Add hooks to get intermediate outcomes
        self.decoder_outs = {}
        for i, layer in enumerate(self.t_decoder.layers):
            name = f"layer_{i}"
            layer.register_forward_hook(get_hook(self.decoder_outs, name))

        self.kpt_extractor_name = kpt_extractor_name
        self.extractor = load_extractor(kpt_extractor_name)

    def forward(self, x, intrinsics=None, depth=None, mask=None):
        if mask is not None:
            x = x * mask
        bs, c, h, w = x.shape
        if bs > 1:
            extracted_kpts = [self.extractor.extract(x[i : i + 1]) for i in range(bs)]
            extracted_kpts = [{k: v[0] for k, v in kpts.items()} for kpts in extracted_kpts]
            # pad with zeros up to the max number of keypoints
            max_kpts = max([len(kpts["keypoints"]) for kpts in extracted_kpts])
            extracted_kpts = copy.deepcopy(extracted_kpts)
            for kpts in extracted_kpts:
                for k in kpts.keys():
                    pad_len = max_kpts - len(kpts[k])
                    if pad_len > 0:
                        if k in ["keypoints", "descriptors"]:
                            kpts[k] = F.pad(kpts[k], (0, 0, 0, pad_len), value=0)
            extracted_kpts = {
                k: torch.stack([v[k] for v in extracted_kpts], dim=0) for k in ["keypoints", "descriptors"]
            }
        else:
            extracted_kpts = self.extractor.extract(x)

        descriptors = extracted_kpts["descriptors"]
        kpt_pos = extracted_kpts["keypoints"]

        # TODO: check kpt norm
        kpt_pos = kpt_pos / torch.tensor([w, h], dtype=kpt_pos.dtype).to(kpt_pos.device)

        if depth is not None:
            assert intrinsics is not None
            # get depth_1d by sampling depth map at kpt_pos as int (ignoring zero kpt pos)
            raise NotImplementedError("Need to implement depth sampling")
            # TODO: get mask of padded keypoints
            kpt_pos = backproj_2d_to_3d_batch(kpt_pos, depth=depth, K=intrinsics)
        elif intrinsics is not None:
            kpt_pos = calibrate_2d_pts_batch(kpt_pos, K=intrinsics)
        tokens = descriptors
        tokens = self.conv1x1(tokens.transpose(-1, -2))
        tokens = rearrange(tokens, "b c hw -> b hw c")

        pos_enc = self.pe_encoder(kpt_pos).to(tokens.device)
        out_encoder = self.t_encoder(tokens + pos_enc)

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
