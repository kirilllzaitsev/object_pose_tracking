import functools
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(chan_in, chan_out):
    return nn.Sequential(
        nn.Conv2d(chan_in, chan_out, 3, padding=1),
        nn.LeakyReLU(),
        nn.Conv2d(chan_out, chan_out, 3, padding=1),
        nn.LeakyReLU(),
    )


class DownBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))

        self.net = double_conv(input_channels, filters)
        self.down = nn.Conv2d(filters, filters, 3, padding=1, stride=2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        unet_res = x

        if self.down is not None:
            x = self.down(x)

        x = x + res
        return x, unet_res


class UpBlock(nn.Module):
    def __init__(self, input_channels, filters, use_skip=False):
        super().__init__()
        self.conv_res = nn.ConvTranspose2d(input_channels // 2, filters, 1, stride=2)
        if not use_skip:
            input_channels = input_channels // 2
        self.net = double_conv(input_channels, filters)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.input_channels = input_channels
        self.filters = filters

    def forward(self, x, res):
        *_, h, w = x.shape
        conv_res = self.conv_res(x, output_size=(h * 2, w * 2))
        x = self.up(x)
        if res is not None:
            res = F.interpolate(res, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
            x = torch.cat((x, res), dim=1)
        x = self.net(x)
        x = x + conv_res
        return x


class UNet(nn.Module):
    def __init__(self, network_capacity=8, num_init_filters=1, num_layers=8, fmap_max=512, use_skip=True):
        super().__init__()

        filters = [num_init_filters] + [(network_capacity) * (2**i) for i in range(num_layers + 1)]

        set_fmap_max = functools.partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        filters[-1] = filters[-2]

        chan_in_out = list(zip(filters[:-1], filters[1:]))
        chan_in_out = list(map(list, chan_in_out))

        down_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DownBlock(in_chan, out_chan, downsample=is_not_last)
            down_blocks.append(block)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)

        last_chan = filters[-1]

        self.conv = double_conv(last_chan, last_chan)

        dec_chan_in_out = chan_in_out[:-1][::-1]
        self.use_skip = use_skip
        self.up_blocks = nn.ModuleList(list(map(lambda c: UpBlock(c[1] * 2, c[0], use_skip=use_skip), dec_chan_in_out)))

    def forward(self, x):
        b, c, h, w = x.shape
        ress = []
        for down_block in self.down_blocks:
            x, res = down_block(x)
            if self.use_skip:
                ress.append(res)
            else:
                ress.append(None)

        mid = self.conv(x) + x
        mid_pooled = F.adaptive_avg_pool2d(mid, 1).view(b, -1)
        x = mid

        for res, up_block in zip(ress[:-1][::-1], self.up_blocks):
            x = up_block(x, res=res)

        dec_out = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        dec_out = dec_out.sigmoid()
        return dec_out, mid_pooled


if __name__ == "__main__":
    model = UNet(network_capacity=32, num_init_filters=1, num_layers=6, fmap_max=512, use_skip=False).cuda()
    x = torch.randn(1, 1, 480, 640).cuda()
    y = model(x)
    print(y.shape)
