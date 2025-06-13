import copy

import einops
import torch
import torch.nn as nn
from einops import rearrange
from pose_tracking.models.cnnlstm import MLP, MLPFactors
from pose_tracking.models.encoders import get_encoders
from pose_tracking.models.matcher import box_cxcywh_to_xyxy
from pose_tracking.models.pos_encoding import (
    PosEncoding,
    PosEncodingCoord,
    PosEncodingDepth,
    PositionEmbeddingLearned,
    SpatialPosEncoding,
    timestep_embedding,
)
from pose_tracking.utils.detr_utils import get_crops
from pose_tracking.utils.geom import backproj_2d_pts, calibrate_2d_pts_batch
from pose_tracking.utils.kpt_utils import (
    extract_kpts,
    get_kpt_within_mask_indicator,
    load_extractor,
)
from pose_tracking.utils.misc import init_params, print_cls
from pose_tracking.utils.segm_utils import mask_morph
from timm.models.resnet import resnet18, resnet50
from torchvision.models import resnet18, resnet50


def get_hook(outs, name):
    def hook(self, input, output):
        outs[name] = output

    return hook


class CNNFeatureExtractor(nn.Module):
    def __init__(self, out_dim=256, model_name="resnet18"):
        super().__init__()
        if model_name == "resnet18":
            self.model = resnet18(pretrained=True)
            fc_in_dim = 512
        elif model_name == "resnet50":
            self.model = resnet50(pretrained=True)
            fc_in_dim = 2048
        else:
            raise ValueError(f"Unknown model name {model_name}")

        if out_dim == fc_in_dim:
            self.model.fc = nn.Identity()
        else:
            self.model.fc = nn.Linear(fc_in_dim, out_dim)

    def forward(self, x):
        x = self.model(x)
        return x


class FactorTransformer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=1,
        )

        self.heads = {
            k: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            )
            for k in ["rot", "t"]
        }
        self.heads = nn.ModuleDict(self.heads)

    def forward(self, obs_tokens, factor_tokens):
        decoded = self.decoder(factor_tokens, obs_tokens)
        out = {}
        decoded_rot = decoded[:, 0]
        decoded_t = decoded[:, 1]
        out["decoded"] = decoded
        out["rot"] = self.heads["rot"](decoded_rot).squeeze(-1)
        out["t"] = self.heads["t"](decoded_t).squeeze(-1)
        return out


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
        do_refinement_with_attn=False,
        use_depth=False,
        final_feature_dim=None,
        pose_token_time_encoding="sin",
        factors=None,
        roi_feature_dim=256,
        do_extract_rt_features=False,
        use_v1_code=False,
        use_uncertainty=False,
        use_render_token=False,
    ):
        super().__init__()

        self.use_pose_tokens = use_pose_tokens
        self.use_roi = use_roi
        self.use_depth = use_depth
        self.do_refinement_with_attn = do_refinement_with_attn
        self.do_extract_rt_features = do_extract_rt_features
        self.use_v1_code = use_v1_code
        self.use_uncertainty = use_uncertainty
        self.use_render_token = use_render_token

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
        self.factors = factors
        self.roi_feature_dim = roi_feature_dim

        self.use_rot = not opt_only or (opt_only and "rot" in opt_only)
        self.use_t = not opt_only or (opt_only and "t" in opt_only)
        self.use_boxes = not opt_only or (opt_only and "boxes" in opt_only)
        self.do_predict_2d_t = t_out_dim == 2
        self.use_factors = factors is not None
        if self.use_factors:
            self.global_factors = [f for f in factors if f in ["scale"]]
            self.local_factors = [f for f in factors if f not in self.global_factors]

        self.pe_encoder = self.get_pos_encoder(encoding_type, n_tokens=n_tokens * (2 if use_depth else 1))

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
        if self.use_boxes:
            self.bbox_mlps = get_clones(
                MLP(
                    in_dim=d_model,
                    out_dim=4,
                    hidden_dim=head_hidden_dim,
                    num_layers=3,
                    dropout=dropout_heads,
                ),
                n_layers,
            )
        if self.use_t:
            t_mlp_in_dim = d_model
            self.t_mlp_in_dim = t_mlp_in_dim
            self.t_mlps = get_clones(
                MLP(
                    t_mlp_in_dim,
                    t_out_dim,
                    hidden_dim=d_model if use_v1_code else head_hidden_dim,
                    num_layers=head_num_layers,
                    dropout=dropout_heads,
                    do_return_last_latent=do_extract_rt_features,
                ),
                n_layers,
            )
        if self.use_rot:
            rot_mlp_in_dim = d_model
            if use_roi:
                rot_mlp_in_dim += roi_feature_dim
            self.rot_mlp_in_dim = rot_mlp_in_dim
            self.rot_mlps = get_clones(
                MLP(
                    rot_mlp_in_dim,
                    rot_out_dim,
                    hidden_dim=d_model if use_v1_code else head_hidden_dim,
                    num_layers=head_num_layers,
                    dropout=dropout_heads,
                    do_return_last_latent=do_extract_rt_features,
                ),
                n_layers,
            )
        if self.do_predict_2d_t:
            self.depth_embed = get_clones(
                MLP(
                    in_dim=d_model,
                    out_dim=1,
                    hidden_dim=d_model if use_v1_code else head_hidden_dim,
                    num_layers=head_num_layers,
                    dropout=dropout_heads,
                ),
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
            if pose_token_time_encoding == "learned":
                max_seq_len = 3  # consider tokens from n last timesteps
                self.time_pos_encoder = self.get_pos_encoder("learned", n_tokens=max_seq_len)
        if do_refinement_with_attn:
            pose_refiner_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
            )
            pose_refiner_n_layers = 1
            self.pose_refiner_transformer = nn.TransformerEncoder(
                pose_refiner_encoder_layer, num_layers=pose_refiner_n_layers
            )

        # Add hooks to get intermediate outcomes
        self.decoder_outs = {}
        for i, layer in enumerate(self.t_decoder.layers):
            name = f"layer_{i}"
            layer.register_forward_hook(get_hook(self.decoder_outs, name))

        if self.use_roi:
            self.roi_cnn = CNNFeatureExtractor(out_dim=roi_feature_dim, model_name="resnet50")

        if self.use_factors:
            assert len(self.global_factors + self.local_factors) > 0
            if len(self.local_factors) > 0:
                self.crop_cnn = CNNFeatureExtractor(out_dim=roi_feature_dim, model_name="resnet18")
                for p in self.crop_cnn.parameters():
                    p.requires_grad = False
            self.factor_mlps = {}
            for k in factors:
                in_dim = roi_feature_dim if k in self.local_factors else d_model
                self.factor_mlps[k] = get_clones(
                    MLPFactors(in_dim, 10, d_model, num_layers=2, dropout=0.2, act_out=None),
                    n_layers,
                )
            self.factor_mlps = nn.ModuleDict(self.factor_mlps)
        # if self.use_render_token:
        #     self.render_cnn = CNNFeatureExtractor(out_dim=roi_feature_dim, model_name="resnet18")
        if self.use_uncertainty:
            self.n_free_factors = 2  # cover other factors
            self.n_uncertainty_tokens = 2  # rot/t
            self.free_factors = nn.Parameter(
                torch.rand((1, self.n_queries, d_model, self.n_free_factors)), requires_grad=True
            )
            self.uncertainty_tokens = nn.Parameter(
                torch.rand((1, self.n_queries, d_model, self.n_uncertainty_tokens)), requires_grad=True
            )
            self.uncertainty_layer = get_clones(FactorTransformer(d_model, n_heads, dropout=dropout), n_layers)

        init_params(self)

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

    def forward(self, x, pose_tokens=None, prev_tokens=None, depth=None, pose_renderer_fn=None, **kwargs):
        extract_res = self.extract_tokens(x, depth=depth, **kwargs)
        tokens = extract_res["tokens"]

        if self.encoding_type == "learned":
            pos_enc = self.pe_encoder(extract_res["img_features"])
            pos_enc = rearrange(pos_enc, "b c h w -> b (h w) c")
            if self.use_depth:
                pos_enc_depth = self.pe_encoder(extract_res["depth_features"])
                pos_enc_depth = rearrange(pos_enc_depth, "b c h w -> b (h w) c")
                pos_enc = torch.cat([pos_enc, pos_enc_depth], dim=1)
        elif self.encoding_type == "sin":
            pos_enc = self.pe_encoder(tokens)
            if self.use_depth:
                pos_enc_depth = self.pe_encoder(extract_res["tokens_depth"])
                pos_enc = torch.cat([pos_enc, pos_enc_depth], dim=0)
        else:
            raise ValueError(f"Unknown encoding type {self.encoding_type}")

        if self.use_depth:
            tokens = torch.cat([tokens, extract_res["tokens_depth"]], dim=-1)

        return self.forward_tokens(
            tokens,
            pos_enc,
            prev_pose_tokens=pose_tokens,
            prev_tokens=prev_tokens,
            img_features=extract_res.get("img_features"),
            rgb=x,
            pose_renderer_fn=pose_renderer_fn,
        )

    def forward_tokens(
        self,
        tokens,
        pos_enc,
        memory_key_padding_mask=None,
        prev_pose_tokens=None,
        prev_tokens=None,
        img_features=None,
        rgb=None,
        pose_renderer_fn=None,
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
            out = {
                "pred_logits": pred_logits,
            }
            if self.use_boxes:
                pred_boxes = self.bbox_mlps[layer_idx](o)
                pred_boxes = torch.sigmoid(pred_boxes)
                out["pred_boxes"] = pred_boxes
            if self.use_pose_tokens:
                pose_token = self.pose_proj(o)
                if prev_pose_tokens is not None and len(prev_pose_tokens) > 0:
                    # time_pos_embed = sinusoidal_embedding(len(prev_pose_tokens), self.d_model)
                    prev_pose_tokens_layer = prev_pose_tokens[layer_idx]
                    b, num_prev_tokens, *_ = prev_pose_tokens_layer.shape
                    time_pos_embed = timestep_embedding(torch.arange(num_prev_tokens + 1).to(o.device), self.d_model)
                    pose_tokens = torch.cat([prev_pose_tokens_layer, pose_token.unsqueeze(1)], dim=1)
                    pose_tokens = einops.rearrange(pose_tokens, "b t q d -> (b q) t d")
                    pose_tokens_pe = pose_tokens + time_pos_embed.unsqueeze(0)
                    mask = self.get_causal_attn_mask_for_pose_tokens(pose_tokens)
                    pose_tokens_enc = self.pose_token_transformer(pose_tokens_pe, mask=mask)
                    pose_tokens_enc = einops.rearrange(pose_tokens_enc, "(b q) t d -> b t q d", b=b)
                    pose_token = pose_tokens_enc[:, -1]
            else:
                pose_token = o

            if self.do_refinement_with_attn:
                # uses pose tokens from prev layers, pos-embedded in time
                num_prev_tokens = layer_idx
                time_pos_embed = timestep_embedding(torch.arange(num_prev_tokens + 1).to(o.device), self.d_model)
                prev_layers_pose_tokens = torch.stack([o["pose_token"] for o in outs[:layer_idx]] + [pose_token], dim=1)
                layers_pose_tokens_pe = prev_layers_pose_tokens + time_pos_embed.unsqueeze(1)
                layers_pose_tokens = einops.rearrange(layers_pose_tokens_pe, "b l q d -> (b q) l d")
                mask = self.get_causal_attn_mask_for_pose_tokens(layers_pose_tokens)
                pose_layers_tokens_enc = self.pose_refiner_transformer(layers_pose_tokens, mask=mask)
                pose_layers_tokens_enc = einops.rearrange(
                    pose_layers_tokens_enc, "(b q) l d -> b l q d", b=pose_token.shape[0], d=self.d_model
                )
                pose_token = pose_layers_tokens_enc[:, -1]

            t_mlp_in = pose_token.clone()
            if self.use_roi:
                assert rgb is not None
                hw = rgb.shape[-2:]
                pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
                rgb_crop = get_crops(rgb, pred_boxes_xyxy, hw=hw, crop_size=(60 * 2, 80 * 2), padding=5)
                crop_feats = self.roi_cnn(rgb_crop)
                crop_feats = rearrange(crop_feats, "(b q) d -> b q d", q=self.n_queries)
                # TODO: check allocentric
                rot_mlp_in = torch.cat([pose_token, crop_feats], dim=-1)
            else:
                rot_mlp_in = pose_token
            if self.use_rot:
                if self.do_extract_rt_features:
                    pred_rot, last_latent_rot = self.rot_mlps[layer_idx](rot_mlp_in)
                    out["last_latent_rot"] = last_latent_rot
                else:
                    pred_rot = self.rot_mlps[layer_idx](rot_mlp_in)
                out["rot"] = pred_rot
            if self.use_t:
                if self.do_extract_rt_features:
                    pred_t, last_latent_t = self.t_mlps[layer_idx](t_mlp_in)
                    out["last_latent_t"] = last_latent_t
                else:
                    pred_t = self.t_mlps[layer_idx](t_mlp_in)
                out["t"] = pred_t
            if self.do_predict_2d_t:
                outputs_depth = self.depth_embed[layer_idx](pose_token)
                out["center_depth"] = outputs_depth

            b = pred_logits.shape[0]
            if self.use_uncertainty:
                factor_tokens = torch.cat(
                    [
                        self.uncertainty_tokens.repeat(b, 1, 1, 1),
                        self.free_factors.repeat(b, 1, 1, 1),
                    ],
                    dim=-1,
                )

                assert rgb is not None
                hw = rgb.shape[-2:]
                pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
                rgb_crop = get_crops(rgb, pred_boxes_xyxy.detach(), hw=hw, crop_size=(80 * 2, 80 * 2))
                crop_feats = self.crop_cnn(rgb_crop)
                if self.use_factors:
                    factors = {}
                    factor_latents = {}
                    if len(self.local_factors) > 0:
                        for factor in self.local_factors:
                            factor_out = self.factor_mlps[factor][layer_idx](crop_feats)
                            factors[factor] = factor_out["out"]
                            factor_latents[factor] = factor_out["last_hidden"]
                        for k, v in factors.items():
                            factors[k] = einops.rearrange(v, "(b q) d -> b q d", b=b, q=self.n_queries)
                        for k, v in factor_latents.items():
                            factor_latents[k] = einops.rearrange(v, "(b q) d -> b q d", b=b, q=self.n_queries)
                    for factor in self.global_factors:
                        factor_out = self.factor_mlps[factor][layer_idx](o.detach())
                        factors[factor] = factor_out["out"]
                        factor_latents[factor] = factor_out["last_hidden"]
                    out["factors"] = factors

                    all_factor_latents = torch.stack([v for v in factor_latents.values()], dim=-1)
                    factor_tokens = torch.cat([factor_tokens, all_factor_latents], dim=-1)
                obs_tokens = o.unsqueeze(-1).detach()
                if self.use_pose_tokens:
                    obs_tokens = torch.cat([obs_tokens, pose_token.unsqueeze(-1).detach()], dim=-1)
                crop_feats_per_q = einops.rearrange(crop_feats, "(b q) c -> b q c", b=b)
                obs_tokens = torch.cat(
                    [
                        obs_tokens,
                        last_latent_rot.unsqueeze(-1).detach(),
                        last_latent_t.unsqueeze(-1).detach(),
                        crop_feats_per_q.unsqueeze(-1).detach(),
                    ],
                    dim=-1,
                )
                if self.use_render_token:
                    assert pose_renderer_fn is not None
                    assert self.n_queries == 1, "rendering works for 1 q for now"
                    render_poses = torch.cat([out["t"], out["rot"]], dim=-1)
                    render_poses = einops.rearrange(render_poses, "b q r -> (b q) r")
                    rendered = pose_renderer_fn(poses_pred=render_poses)
                    # TODO: a learnable net? crop_cnn is frozen atm
                    rendered_feats = self.crop_cnn(rendered["rgb"])
                    rendered_feats = einops.rearrange(rendered_feats, "(b q) d -> b q d", b=b)
                    obs_tokens = torch.cat([obs_tokens, rendered_feats.unsqueeze(-1).detach()], dim=-1)

                obs_tokens = einops.rearrange(obs_tokens, "b q d f -> (b q) f d")
                factor_tokens = einops.rearrange(factor_tokens, "b q d f -> (b q) f d")
                u_out = self.uncertainty_layer[layer_idx](obs_tokens=obs_tokens, factor_tokens=factor_tokens)
                out["decoded"] = einops.rearrange(u_out.pop("decoded"), "(b q) f d -> b q f d", b=b, q=self.n_queries)
                for k, v in u_out.items():
                    v = einops.rearrange(v, "(b q) -> b q", b=b, q=self.n_queries)
                    out[f"uncertainty_{k}"] = v

            out["pose_token"] = pose_token
            outs.append(out)
        last_out = outs.pop()
        res = {
            "pred_logits": last_out["pred_logits"],
            "pred_boxes": last_out.get("pred_boxes"),
            "decoded": last_out.get("decoded"),
            "aux_outputs": outs,
        }
        if self.use_rot:
            res["rot"] = last_out["rot"]
            if self.do_extract_rt_features:
                res["last_latent_rot"] = last_out["last_latent_rot"]
        if self.use_t:
            res["t"] = last_out["t"]
            if self.do_extract_rt_features:
                res["last_latent_t"] = last_out["last_latent_t"]
        if self.do_predict_2d_t:
            res["center_depth"] = last_out["center_depth"]
        if self.use_pose_tokens:
            res["pose_tokens"] = [o["pose_token"] for o in outs] + [last_out["pose_token"]]
        if self.use_factors:
            res["factors"] = last_out["factors"]
        if self.use_uncertainty:
            res["uncertainty_rot"] = last_out["uncertainty_rot"]
            res["uncertainty_t"] = last_out["uncertainty_t"]
        res["tokens"] = tokens_enc

        return res

    def get_causal_attn_mask_for_pose_tokens(self, pose_tokens):
        num_tokens = pose_tokens.shape[1]
        mask = torch.triu(torch.ones((num_tokens, num_tokens)) * float("-inf"), diagonal=1)
        mask = mask.to(pose_tokens.device)
        return mask

    def extract_tokens(self, rgb, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return print_cls(
            self,
            extra_str=super().__repr__(),
            excluded_attrs=["decoder_outs", "backbone_feats", "backbone_rgb_feats", "backbone_depth_feats"],
        )

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

        self.proj = nn.Linear(self.final_feature_dim, self.d_model)
        if self.use_depth:
            self.proj_depth = nn.Linear(self.final_feature_dim, self.d_model)

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

        res = {}
        res["img_features"] = tokens
        tokens = rearrange(tokens, "b c h w -> b (h w) c")

        if self.use_depth:
            _ = self.backbone_depth(depth)
            tokens_depth = self.backbone_depth_feats["layer4"]
            res["depth_features"] = tokens_depth
            tokens_depth = rearrange(tokens_depth, "b c h w -> b (h w) c")
            tokens_depth = self.proj_depth(tokens_depth)
            res["tokens_depth"] = rearrange(tokens_depth, "b n d -> b d n")

        tokens = self.proj(tokens)
        res["tokens"] = rearrange(tokens, "b n d -> b d n")
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
        do_freeze_kpt_detector=True,
        **kwargs,
    ):
        self.do_calibrate_kpt = do_calibrate_kpt
        self.do_freeze_kpt_detector = do_freeze_kpt_detector

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
            self.pe_depth = PosEncodingDepth(self.d_model, use_mlp=False)
        self.proj = nn.Conv1d(self.token_dim, self.d_model, kernel_size=1, stride=1)

        self.kpt_extractor_name = kpt_extractor_name
        self.extractor = load_extractor(kpt_extractor_name)

        if do_freeze_kpt_detector:
            for param in self.extractor.parameters():
                param.requires_grad = False

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
            rgb=x,
        )

        for k in ["kpts", "descriptors", "score_map"]:
            out[k] = extract_res[k]

        return out

    def extract_tokens(self, rgb, intrinsics=None, depth=None, mask=None):
        if self.use_mask_on_input:
            assert mask is not None and len(mask) > 0
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
            K_norm = intrinsics.clone().float()
            K_norm[..., 0, 0] /= w
            K_norm[..., 0, 2] /= w
            K_norm[..., 1, 1] /= h
            K_norm[..., 1, 2] /= h
            if self.do_backproj_kpts_to_3d:
                assert depth is not None
                kpts = backproj_2d_pts(kpts, depth=depth_1d, K=K_norm)
            elif self.do_calibrate_kpt:
                kpts = calibrate_2d_pts_batch(kpts, K=K_norm)

        tokens = descriptors  # B x N x D

        if self.use_mask_as_obj_indicator:
            # TODO: n objs (kpts on each obj)
            obj_indicator = get_kpt_within_mask_indicator(extracted_kpts["keypoints"], mask.squeeze(1)).unsqueeze(1)
            tokens = torch.cat([tokens, obj_indicator.transpose(-1, -2)], dim=-1)
        if self.use_depth:
            depth_1d_pe = self.pe_depth(depth_1d.unsqueeze(-1))
            if self.use_mask_as_obj_indicator:
                depth_1d_pe = torch.cat([depth_1d_pe, torch.zeros_like(obj_indicator).transpose(-2, -1)], dim=-1)
            tokens = tokens + depth_1d_pe

        tokens = self.proj(tokens.transpose(-1, -2))

        return {
            "tokens": tokens,
            "kpts": kpts,
            "score_map": extracted_kpts["score_map"],
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
