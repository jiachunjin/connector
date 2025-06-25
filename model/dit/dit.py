import torch
import torch.nn as nn
from .dit_basic import DiTBlock, TimestepEmbedder, FinalLayer, precompute_freqs_cis_2d, LabelEmbedder


class DiTClassConditional(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.depth = config.depth
        self.x_dim = config.x_dim
        self.grid_size = config.grid_size
        self.num_classes = config.num_classes

        self.x_proj = nn.Linear(self.x_dim, self.hidden_size)
        self.y_embedder = LabelEmbedder(config.num_classes, config.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_size, self.num_heads) for _ in range(self.depth)
        ])
        self.final_layer = FinalLayer(self.hidden_size, self.x_dim)
        self.precompute_pos = dict()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_heads, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def forward(self, x_t, y, t):
        B, L, d = x_t.shape
        pos = self.fetch_pos(self.grid_size, self.grid_size, x_t.device)
        x = self.x_proj(x_t)
        # if self.training:
        #     y = torch.where(torch.rand(y.shape, device=y.device, dtype=torch.float32) < 0.1, torch.full_like(y, self.num_classes), y)
            
        y = self.y_embedder(y)
        t = self.t_embedder(t, x_t.dtype)
        c = t + y
        for i, block in enumerate(self.blocks):
            x = block(x, c, pos, None)
        
        x = self.final_layer(x, c)
        
        return x


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("config/dit_on_siglip_dim_down.yaml")
    dit = DiTClassConditional(config.dit)
    x_t = torch.randn(1, 576, 32)
    y = torch.randint(0, 1000, (1,))
    t = torch.randint(0, 1000, (1,))
    x = dit(x_t, y, t)
    print(x.shape)