import torch
import torch.nn as nn


class PosEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2).float()

        self.encoding = torch.zeros(max_len, d_model)
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        B, D, L = x.size()
        return self.encoding[:L, :]

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.encoding = self.encoding.to(*args, **kwargs)
        return self


class SpatialPosEncoding(nn.Module):
    def __init__(self, d_model, ndim=2, enc_type="mlp"):
        super().__init__()
        self.ndim = ndim
        if enc_type == "mlp":
            self.px_pos_to_encoding = nn.Linear(ndim, d_model)
        else:
            raise ValueError(f"Unknown enc_type: {enc_type}")

    def forward(self, x):
        enc = self.px_pos_to_encoding(x)
        return enc


def timestep_embedding(timesteps, dim, max_period=100):
    """https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/nn.py#L103
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
