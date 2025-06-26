import torch
import torch.nn as nn
from pose_tracking.models.detr import DETRBase
from pose_tracking.models.pos_encoding import (
    PosEncodingCoord,
    PosEncodingDepth,
    SpatialPosEncoding,
)
from pose_tracking.utils.geom import backproj_2d_pts, calibrate_2d_pts_batch
from pose_tracking.utils.kpt_utils import (
    extract_kpts,
    get_kpt_within_mask_indicator,
    load_extractor,
)
from pose_tracking.utils.segm_utils import mask_morph


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
