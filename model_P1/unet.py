import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class UNet(nn.Module):
    """
    Simplified U-Net for DDPM.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 128,
        num_res_blocks: int = 2,
    ):
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )

        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Down
        self.down1 = nn.ModuleList([ResBlock(model_channels, model_channels, model_channels * 4) for _ in range(num_res_blocks)])
        self.down2 = nn.Conv2d(model_channels, model_channels * 2, 4, 2, 1)
        self.down3 = nn.ModuleList([ResBlock(model_channels * 2, model_channels * 2, model_channels * 4) for _ in range(num_res_blocks)])
        self.down4 = nn.Conv2d(model_channels * 2, model_channels * 4, 4, 2, 1)
        self.down5 = nn.ModuleList([ResBlock(model_channels * 4, model_channels * 4, model_channels * 4) for _ in range(num_res_blocks)])

        # Middle
        self.middle = nn.ModuleList([
            ResBlock(model_channels * 4, model_channels * 4, model_channels * 4),
            ResBlock(model_channels * 4, model_channels * 4, model_channels * 4),
        ])

        # Up
        self.up1 = nn.ConvTranspose2d(model_channels * 4, model_channels * 2, 4, 2, 1)
        self.up2 = nn.ModuleList([ResBlock(model_channels * 4, model_channels * 4, model_channels * 4) for _ in range(num_res_blocks)])
        self.up2_conv = nn.Conv2d(model_channels * 4, model_channels * 2, 1)
        self.up3 = nn.ConvTranspose2d(model_channels * 2, model_channels, 4, 2, 1)
        self.up4 = nn.ModuleList([ResBlock(model_channels * 2, model_channels * 2, model_channels * 4) for _ in range(num_res_blocks)])
        self.up4_conv = nn.Conv2d(model_channels * 2, model_channels, 1)

        self.conv_out = nn.Conv2d(model_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(timestep_embedding(t, self.conv_in.out_channels))

        x1 = self.conv_in(x)
        for layer in self.down1:
            x1 = layer(x1, t_emb)

        x2 = self.down2(x1)
        for layer in self.down3:
            x2 = layer(x2, t_emb)

        x3 = self.down4(x2)
        for layer in self.down5:
            x3 = layer(x3, t_emb)

        for layer in self.middle:
            x3 = layer(x3, t_emb)

        x4 = self.up1(x3)
        x4 = torch.cat([x4, x2], dim=1)
        for layer in self.up2:
            x4 = layer(x4, t_emb)
        x4 = self.up2_conv(x4)

        x5 = self.up3(x4)
        x5 = torch.cat([x5, x1], dim=1)
        for layer in self.up4:
            x5 = layer(x5, t_emb)
        x5 = self.up4_conv(x5)

        return self.conv_out(x5)


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_emb = nn.Linear(time_embed_dim, out_channels)
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + self.time_emb(F.silu(t_emb))[:, :, None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)