import copy
import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from pose_tracking.models.cnnlstm import MLP
from pose_tracking.models.encoders import FrozenBatchNorm2d, get_encoders
from pose_tracking.models.matcher import box_cxcywh_to_xyxy
from pose_tracking.models.pos_encoding import (
    DepthPositionalEncoding,
    PosEncoding,
    PosEncodingCoord,
    PositionEmbeddingLearned,
    SpatialPosEncoding,
    sinusoidal_embedding,
)
from pose_tracking.utils.geom import (
    backproj_2d_to_3d,
    backproj_2d_to_3d_batch,
    calibrate_2d_pts_batch,
)
from pose_tracking.utils.kpt_utils import (
    extract_kpts,
    get_kpt_within_mask_indicator,
    load_extractor,
)
from pose_tracking.utils.misc import print_cls
from pose_tracking.utils.segm_utils import mask_morph
from torchvision.models import resnet18, resnet50, resnet101
from torchvision.ops import roi_align


def get_hook(outs, name):
    def hook(self, input, output):
        outs[name] = output

    return hook


class DETRPretrained(nn.Module):

    def __init__(
        self,
        num_classes,
        use_pretrained_backbone=True,
        rot_out_dim=4,
        t_out_dim=3,
        opt_only=[],
        d_model=256,
        n_layers=6,
        dropout=0.0,
        dropout_heads=0.0,
        head_num_layers=2,
    ):
        super().__init__()

        self.use_pretrained_backbone = use_pretrained_backbone
        self.rot_out_dim = rot_out_dim
        self.t_out_dim = t_out_dim
        self.opt_only = opt_only
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.dropout_heads = dropout_heads
        self.head_num_layers = head_num_layers

        self.use_rot = not opt_only or (opt_only and "rot" in opt_only)
        self.use_t = not opt_only or (opt_only and "t" in opt_only)

        self.model = torch.hub.load("facebookresearch/detr:main", "detr_resnet50", pretrained=use_pretrained_backbone)

        self.class_embed = get_clones(nn.Linear(256, num_classes + 1), n_layers)
        self.bbox_embed = get_clones(
            MLP(d_model, 4, hidden_dim=d_model, num_layers=head_num_layers, dropout=dropout_heads), n_layers
        )
        self.model.class_embed = nn.Identity()
        self.model.bbox_embed = nn.Identity()
        self.model.transformer.decoder.norm = nn.Identity()

        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout

        if self.use_t:
            self.t_mlps = get_clones(
                MLP(d_model, t_out_dim, d_model, num_layers=head_num_layers, dropout=dropout_heads), n_layers
            )
        if self.use_rot:
            self.rot_mlps = get_clones(
                MLP(d_model, rot_out_dim, d_model, num_layers=head_num_layers, dropout=dropout_heads), n_layers
            )

        self.decoder_outs = {}
        for i, layer in enumerate(self.model.transformer.decoder.layers):
            name = f"layer_{i}"
            layer.register_forward_hook(get_hook(self.decoder_outs, name))

    def forward(self, x):
        main_out = self.model(x)
        outs = []
        for layer_idx, (n, o) in enumerate(sorted(self.decoder_outs.items())):
            out = {}
            out["pred_logits"] = self.class_embed[layer_idx](o).transpose(0, 1)
            out["pred_boxes"] = self.bbox_embed[layer_idx](o).sigmoid().transpose(0, 1)
            if self.use_rot:
                pred_rot = self.rot_mlps[layer_idx](o)
                out["rot"] = pred_rot.transpose(0, 1)
            if self.use_t:
                pred_t = self.t_mlps[layer_idx](o)
                out["t"] = pred_t.transpose(0, 1)

            outs.append(out)
        last_out = outs.pop()
        res = {
            "pred_logits": last_out["pred_logits"],
            "pred_boxes": last_out["pred_boxes"],
            "aux_outputs": outs,
        }
        if self.use_rot:
            res["rot"] = last_out["rot"]
        if self.use_t:
            res["t"] = last_out["t"]

        return res


class DETRBase(nn.Module):

    def __init__(
        self,
        num_classes,
        d_model=256,
        n_tokens=15 * 20,
        n_layers=6,
        n_heads=8,
        n_queries=100,
        head_hidden_dim=256,
        head_num_layers=2,
        encoding_type="learned",
        opt_only=[],
        dropout=0.0,
        dropout_heads=0.0,
        rot_out_dim=4,
        t_out_dim=3,
        use_pose_tokens=False,
        use_roi=False,
        use_depth=False,
        final_feature_dim=None,
        pose_token_time_encoding="sin",
    ):
        super().__init__()

        self.use_pose_tokens = use_pose_tokens
        self.use_roi = use_roi
        self.use_depth = use_depth

        self.num_classes = num_classes
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.encoding_type = encoding_type
        self.dropout = dropout
        self.dropout_heads = dropout_heads
        self.rot_out_dim = rot_out_dim
        self.t_out_dim = t_out_dim
        self.opt_only = opt_only
        self.head_hidden_dim = head_hidden_dim
        self.head_num_layers = head_num_layers
        self.pose_token_time_encoding = pose_token_time_encoding
        self.final_feature_dim = final_feature_dim

        self.use_rot = not opt_only or (opt_only and "rot" in opt_only)
        self.use_t = not opt_only or (opt_only and "t" in opt_only)
        self.do_predict_2d_t = t_out_dim == 2
        self.pe_encoder = self.get_pos_encoder(encoding_type, n_tokens=n_tokens)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )

        self.t_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.queries = nn.Parameter(torch.rand((1, n_queries, d_model)), requires_grad=True)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True, dropout=dropout
        )

        self.t_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        num_classes_bg = num_classes + 1
        self.class_mlps = get_clones(
            MLP(
                in_dim=d_model, out_dim=num_classes_bg, hidden_dim=head_hidden_dim, num_layers=1, dropout=dropout_heads
            ),
            n_layers,
        )
        self.bbox_mlps = get_clones(
            MLP(
                in_dim=d_model, out_dim=4, hidden_dim=head_hidden_dim, num_layers=head_num_layers, dropout=dropout_heads
            ),
            n_layers,
        )
        if self.use_t:
            self.t_mlps = get_clones(MLP(d_model, t_out_dim, d_model, head_num_layers, dropout=dropout_heads), n_layers)
        if self.use_rot:
            rot_mlp_in_dim = d_model
            if use_roi:
                rot_mlp_in_dim += d_model
            self.rot_mlps = get_clones(
                MLP(rot_mlp_in_dim, rot_out_dim, d_model, head_num_layers, dropout=dropout_heads), n_layers
            )
        if self.do_predict_2d_t:
            self.depth_embed = get_clones(
                MLP(in_dim=d_model, out_dim=1, hidden_dim=d_model, num_layers=head_num_layers, dropout=dropout_heads),
                n_layers,
            )
        if use_pose_tokens:
            self.pose_proj = nn.Linear(d_model, d_model)
            pose_token_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
            )
            pose_token_n_layers = 1
            self.pose_token_transformer = nn.TransformerEncoder(
                pose_token_encoder_layer, num_layers=pose_token_n_layers
            )
            # TODO
            if pose_token_time_encoding == "learned":
                seq_len = 3
                self.time_pos_encoder = self.get_pos_encoder("learned", n_tokens=seq_len)

        # Add hooks to get intermediate outcomes
        self.decoder_outs = {}
        for i, layer in enumerate(self.t_decoder.layers):
            name = f"layer_{i}"
            layer.register_forward_hook(get_hook(self.decoder_outs, name))

        if use_roi:
            assert final_feature_dim is not None
            self.rgb_roi_cnn = nn.Sequential(
                nn.Conv2d(final_feature_dim, head_hidden_dim, kernel_size=3, padding=1, stride=1),
                FrozenBatchNorm2d(head_hidden_dim),
                nn.ReLU(),
                nn.Conv2d(head_hidden_dim, head_hidden_dim, kernel_size=3, padding=1, stride=1),
                FrozenBatchNorm2d(head_hidden_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

    def get_pos_encoder(self, encoding_type, sin_max_len=1024, n_tokens=None):
        if encoding_type == "learned":
            assert n_tokens is not None
            pe_encoder = PositionEmbeddingLearned(num_pos_feats=self.d_model // 2)
        elif encoding_type == "sin":
            # tweak sin_max_len=1024 for img-based pe
            pe_encoder = PosEncoding(self.d_model, max_len=sin_max_len)
        elif encoding_type == "none":
            pe_encoder = None
        else:
            raise ValueError(f"Unknown encoding type {encoding_type}")
        return pe_encoder

    def forward(self, x, pose_tokens=None, prev_tokens=None, depth=None, **kwargs):
        extract_res = self.extract_tokens(x, depth=depth, **kwargs)
        tokens = extract_res["tokens"]

        if self.encoding_type == "learned":
            pos_enc = self.pe_encoder(extract_res["img_features"])
            pos_enc = rearrange(pos_enc, "b c h w -> b (h w) c")
        elif self.encoding_type == "sin":
            pos_enc = self.pe_encoder(tokens)
        else:
            raise ValueError(f"Unknown encoding type {self.encoding_type}")

        return self.forward_tokens(
            tokens,
            pos_enc,
            prev_pose_tokens=pose_tokens,
            prev_tokens=prev_tokens,
            img_features=extract_res.get("img_features"),
        )

    def forward_tokens(
        self, tokens, pos_enc, memory_key_padding_mask=None, prev_pose_tokens=None, prev_tokens=None, img_features=None
    ):
        tokens = tokens.transpose(-1, -2)  # (B, D, N) -> (B, N, D)

        tokens_enc = self.t_encoder(tokens + pos_enc, src_key_padding_mask=memory_key_padding_mask)

        if prev_tokens is None:
            # unclear if have to concat with itself as in trackformer
            tokens_dec = tokens_enc
            prev_tokens = tokens_enc
        else:
            tokens_dec = torch.cat([tokens_enc, prev_tokens], dim=1)

        queries_dec = self.t_decoder(
            self.queries.repeat(len(tokens_dec), 1, 1), tokens_dec, memory_key_padding_mask=memory_key_padding_mask
        )

        outs = []
        for layer_idx, (n, o) in enumerate(sorted(self.decoder_outs.items())):
            pred_logits = self.class_mlps[layer_idx](o)
            pred_boxes = self.bbox_mlps[layer_idx](o)
            pred_boxes = torch.sigmoid(pred_boxes)
            out = {
                "pred_logits": pred_logits,
                "pred_boxes": pred_boxes,
            }
            if self.use_pose_tokens:
                pose_token = self.pose_proj(o)
                if prev_pose_tokens is not None and len(prev_pose_tokens) > 0:
                    # time_pos_embed = sinusoidal_embedding(len(prev_pose_tokens), self.d_model)
                    prev_pose_tokens_layer = prev_pose_tokens[layer_idx]
                    num_prev_tokens, b, *_ = prev_pose_tokens_layer.shape
                    time_pos_embed = torch.cat(
                        [sinusoidal_embedding(i, self.d_model) for i in range(num_prev_tokens + 1)], dim=0
                    ).to(pose_token.device)
                    prev_pose_tokens_pos = prev_pose_tokens_layer + time_pos_embed[:-1].unsqueeze(1).unsqueeze(1)
                    prev_pose_tokens_pos = einops.rearrange(prev_pose_tokens_pos, "t b q d -> b (t q) d")
                    pose_tokens = torch.cat([prev_pose_tokens_pos, pose_token + time_pos_embed[-1:]], dim=1)
                    pose_tokens_enc = self.pose_token_transformer(pose_tokens)
                    pose_token = pose_tokens_enc[:, -self.n_queries :]
            else:
                pose_token = o

            if self.use_roi:
                assert img_features is not None
                pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
                ind = torch.arange(pred_boxes_xyxy.shape[0]).unsqueeze(1)
                ind = ind.type_as(pred_boxes_xyxy)
                pred_boxes_xyxy_roi = (
                    torch.cat((ind.unsqueeze(1).repeat(1, self.n_queries, 1), pred_boxes_xyxy), dim=-1)
                    .float()
                    .view(-1, 5)
                )
                roi_features = roi_align(img_features, pred_boxes_xyxy_roi, output_size=(7, 7), spatial_scale=1.0)
                roi_features_cnn = self.rgb_roi_cnn(roi_features)
                bs = ind.shape[0]
                roi_features_cnn = roi_features_cnn.view(bs, self.n_queries, -1)
                rot_mlp_in = torch.cat([pose_token, roi_features_cnn], dim=-1)
                out["rot"] = self.rot_mlps[layer_idx](rot_mlp_in)
                out["t"] = self.t_mlps[layer_idx](pose_token)
            else:
                if self.use_rot:
                    pred_rot = self.rot_mlps[layer_idx](pose_token)
                    out["rot"] = pred_rot
                if self.use_t:
                    pred_t = self.t_mlps[layer_idx](pose_token)
                    out["t"] = pred_t
            if self.do_predict_2d_t:
                outputs_depth = self.depth_embed[layer_idx](pose_token)
                out["center_depth"] = outputs_depth

            out["pose_token"] = pose_token
            outs.append(out)
        last_out = outs.pop()
        res = {
            "pred_logits": last_out["pred_logits"],
            "pred_boxes": last_out["pred_boxes"],
            "aux_outputs": outs,
        }
        if self.use_rot:
            res["rot"] = last_out["rot"]
        if self.use_t:
            res["t"] = last_out["t"]
        if self.do_predict_2d_t:
            res["center_depth"] = last_out["center_depth"]
        if self.use_pose_tokens:
            res["pose_tokens"] = [o["pose_token"] for o in outs] + [last_out["pose_token"]]
        res["tokens"] = tokens_enc

        return res

    def extract_tokens(self, rgb, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return print_cls(self, extra_str=super().__repr__(), excluded_attrs=["decoder_outs", "backbone_feats"])

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if self.pe_encoder is not None:
            self.pe_encoder = self.pe_encoder.to(*args, **kwargs)
        return self


class DETR(DETRBase):

    def __init__(
        self,
        *args,
        backbone_name="resnet18",
        **kwargs,
    ):
        self.backbone_name = backbone_name

        self.backbone_weights = None
        if backbone_name == "resnet50":
            self.final_layer_name = "layer4"
            self.final_feature_dim = 2048
        elif backbone_name == "resnet18":
            self.final_feature_dim = 512
            self.final_layer_name = "layer4"
        elif backbone_name == "resnet101":
            self.final_feature_dim = 2048
            self.final_layer_name = "layer4"
        else:
            raise ValueError(f"Unknown backbone {backbone_name}")

        super().__init__(
            *args,
            **kwargs,
            final_feature_dim=self.final_feature_dim,
        )

        self.backbone, self.backbone_depth = get_encoders(model_name=backbone_name, norm_layer_type="frozen_bn")
        self.backbone.fc = nn.Identity()
        self.backbone_depth.fc = nn.Identity()
        conv_1_in_dim = self.final_feature_dim

        if self.use_depth:
            conv_1_in_dim += self.final_feature_dim
        else:
            self.backbone_depth = None

        self.conv1x1 = nn.Conv2d(conv_1_in_dim, self.d_model, kernel_size=1, stride=1)

        self.backbone_rgb_feats = {}

        def hook_resnet50_feats(model, inp, out):
            self.backbone_rgb_feats["layer4"] = out

        self.backbone.layer4.register_forward_hook(hook_resnet50_feats)

        if self.use_depth:
            self.backbone_depth_feats = {}

            def hook_resnet50_feats_depth(model, inp, out):
                self.backbone_depth_feats["layer4"] = out

            self.backbone_depth.layer4.register_forward_hook(hook_resnet50_feats_depth)

    def extract_tokens(self, rgb, depth=None):
        _ = self.backbone(rgb)
        tokens = self.backbone_rgb_feats["layer4"]

        if self.use_depth:
            _ = self.backbone_depth(depth)
            tokens_depth = self.backbone_depth_feats["layer4"]
            tokens = torch.cat([tokens, tokens_depth], dim=1)

        res = {}
        if self.use_roi:
            res["img_features"] = tokens
        res["img_features"] = tokens

        tokens = self.conv1x1(tokens)
        tokens = rearrange(tokens, "b c h w -> b c (h w)")
        res["tokens"] = tokens
        return res


class KeypointDETR(DETRBase):

    def __init__(
        self,
        *args,
        kpt_extractor_name="superpoint",
        kpt_spatial_dim=2,
        descriptor_dim=256,
        use_mask_on_input=False,
        use_mask_as_obj_indicator=False,
        do_calibrate_kpt=False,
        **kwargs,
    ):
        self.do_calibrate_kpt = do_calibrate_kpt

        self.use_mask_on_input = use_mask_on_input
        self.use_mask_as_obj_indicator = use_mask_as_obj_indicator
        self.kpt_spatial_dim = kpt_spatial_dim
        self.descriptor_dim = descriptor_dim

        self.do_backproj_kpts_to_3d = self.kpt_spatial_dim == 3

        super().__init__(
            *args,
            **kwargs,
        )

        self.token_dim = descriptor_dim
        if use_mask_as_obj_indicator:
            self.token_dim += 1
        if self.use_depth:
            self.pe_depth = DepthPositionalEncoding(self.d_model)
        self.conv1x1 = nn.Conv1d(self.token_dim, self.d_model, kernel_size=1, stride=1)

        self.kpt_extractor_name = kpt_extractor_name
        self.extractor = load_extractor(kpt_extractor_name)

    def forward(self, x, pose_tokens=None, prev_tokens=None, **kwargs):
        extract_res = self.extract_tokens(x, **kwargs)
        tokens = extract_res["tokens"]
        memory_key_padding_mask = extract_res.get("memory_key_padding_mask")

        if self.encoding_type == "learned":
            pos_enc = self.pe_encoder
        elif self.encoding_type == "sin":
            pos_enc = self.pe_encoder(tokens)
        elif self.encoding_type == "none":
            pos_enc = torch.zeros_like(tokens)[:, 0].unsqueeze(-1)
        else:
            pos_enc = self.pe_encoder(extract_res["kpts"])

        out = self.forward_tokens(
            tokens,
            pos_enc,
            memory_key_padding_mask=memory_key_padding_mask,
            prev_pose_tokens=pose_tokens,
            prev_tokens=prev_tokens,
            img_features=extract_res.get("img_features"),
        )

        for k in ["kpts", "descriptors"]:
            out[k] = extract_res[k]

        return out

    def extract_tokens(self, rgb, intrinsics=None, depth=None, mask=None):
        if self.use_mask_on_input:
            mask = torch.stack([mask_morph(m, op_name="dilate") for m in mask]).unsqueeze(1)
            rgb = rgb * mask
        bs, c, h, w = rgb.shape
        extracted_kpts = extract_kpts(rgb, extractor=self.extractor)
        memory_key_padding_mask = extracted_kpts.get("memory_key_padding_mask")

        descriptors = extracted_kpts["descriptors"]
        kpts = extracted_kpts["keypoints"]

        if depth is not None:
            depth_1d = []
            for i in range(bs):
                depth_1d.append(depth[i, 0, kpts[i, :, 1].long(), kpts[i, :, 0].long()])
            depth_1d = torch.stack(depth_1d, dim=0).to(kpts.device)
        kpts = kpts / torch.tensor([w, h], dtype=kpts.dtype).to(kpts.device)

        if self.do_backproj_kpts_to_3d or self.do_calibrate_kpt:
            assert intrinsics is not None
            assert depth is not None
            K_norm = intrinsics.clone().float()
            K_norm[..., 0] /= w
            K_norm[..., 1] /= h
            K_norm[..., 0, 2] /= w
            K_norm[..., 1, 2] /= h
            if self.do_backproj_kpts_to_3d:
                kpts = backproj_2d_to_3d_batch(kpts, depth=depth_1d, K=K_norm)
            elif self.do_calibrate_kpt:
                kpts = calibrate_2d_pts_batch(kpts, K=K_norm)

        tokens = descriptors  # B x N x D

        if self.use_mask_as_obj_indicator:
            obj_indicator = get_kpt_within_mask_indicator(extracted_kpts["keypoints"], mask)
            tokens = torch.cat([tokens, obj_indicator.transpose(-1, -2)], dim=-1)
        if self.use_depth:
            tokens = tokens + self.pe_depth(depth_1d.unsqueeze(-1))

        tokens = self.conv1x1(tokens.transpose(-1, -2))

        return {
            "tokens": tokens,
            "kpts": kpts,
            "descriptors": descriptors,
            "memory_key_padding_mask": memory_key_padding_mask,
        }

    def get_pos_encoder(self, encoding_type, n_tokens=None):
        if encoding_type == "spatial":
            pe_encoder = SpatialPosEncoding(self.d_model, ndim=self.kpt_spatial_dim)
        elif encoding_type == "sin_coord":
            pe_encoder = PosEncodingCoord(self.d_model)
        else:
            pe_encoder = super().get_pos_encoder(encoding_type, n_tokens=n_tokens)
        return pe_encoder


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
