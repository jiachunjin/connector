import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .vit_pixel_decoder_basic import Block, precompute_freqs_cis_2d, Upsample, CNN_Encoder


class ViTPixelDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.patch_size = config.patch_size
        self.grid_size = config.grid_size
        self.input_dim = config.input_dim
        self.upsample = getattr(config, "upsample", False)
        self.vae_residual = getattr(config, "vae_residual", False)
        if getattr(config, "siglip_feature_dim_down", None) is not None:
            self.siglip_feature_dim_down = config.siglip_feature_dim_down
            self.siglip_feature_proj = nn.Linear(config.siglip_feature_dim, config.siglip_feature_dim_down)
        else:
            self.siglip_feature_dim_down = None

        self.input_proj = nn.Linear(config.input_dim, config.hidden_size)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.blocks = nn.ModuleList([Block(config.hidden_size, config.num_heads) for _ in range(config.depth)])
        self.norm2 = nn.LayerNorm(config.hidden_size)

        if self.vae_residual:
            self.vae_encoder = CNN_Encoder(output_dim=config.residual_dim)

        if self.upsample:
            self.output = nn.Sequential(
                Rearrange("b (h w) c -> b c h w", h=24, w=24),
                Upsample(self.hidden_size, 256),
                Upsample(256, 64),
                Upsample(64, 16),
                Upsample(16, 3),
            )
        else:
            self.proj_1 = nn.Conv2d(self.hidden_size, 16, 1, padding=0, bias=True)
            self.output_proj = nn.Sequential(
                nn.Conv2d(18, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True),
                Rearrange("b (p1 p2 c) h w -> b c (h p1) (w p2)", p1=self.patch_size, p2=self.patch_size)
            )
            self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)
        self.precompute_pos = dict()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_heads, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def forward(self, x_siglip, original_image=None):
        B, L, D = x_siglip.shape
        pos = self.fetch_pos(self.grid_size, self.grid_size, x_siglip.device)
        
        # 初始化x变量
        if self.siglip_feature_dim_down is not None:
            x = self.siglip_feature_proj(x_siglip)
        else:
            x = x_siglip

        x = self.input_proj(x)
        x = self.norm1(x)
        for block in self.blocks:
            x = block(x, pos)
        x = self.norm2(x)
        if self.upsample:
            rec = self.output(x)
        else:
            residual = self.vae_encoder(original_image) # B C_res H W
            # make residual to zero with 0.1 probability
            residual = residual * (torch.rand(1, 1, 1, 1, device=x.device) < 0.1)
            x = x.permute(0, 2, 1).reshape(-1, self.hidden_size, self.grid_size, self.grid_size).contiguous()
            x = self.proj_1(x) # B C H W
            x = torch.cat([x, residual], dim=1) # B C_res + C H W
            x = self.output_proj(x)
            rec = self.conv_out(x)

        return rec

    def get_feature_dim_down(self, x):
        x = self.siglip_feature_proj(x)

        return x

    def decode_feature_dim_down(self, x):
        B, L, D = x.shape
        pos = self.fetch_pos(self.grid_size, self.grid_size, x.device)
        x = self.input_proj(x)
        x = self.norm1(x)
        for block in self.blocks:
            x = block(x, pos)
        x = self.norm2(x)
        if self.upsample:
            x = self.output(x)
        else:
            x = x.permute(0, 2, 1).reshape(B, self.hidden_size, self.grid_size, self.grid_size).contiguous()
            x = self.output_proj(x)
            x = self.conv_out(x)

        return x


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("config/decoder/residual_layer10.yaml")
    decoder = ViTPixelDecoder(config.decoder)
    x = torch.randn(1, 576, 1024)
    ori = torch.randn(1, 3, 384, 384)
    print(decoder(x, ori).shape)