import torch
import torch.nn as nn
from pose_tracking.models.cnnlstm import MLP
from pose_tracking.models.detr_utils import get_clones, get_hook


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
