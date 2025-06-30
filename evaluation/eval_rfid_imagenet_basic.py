import torch.nn as nn

from janus.models import MultiModalityCausalLM
from model.decoder.vit_pixel_decoder import ViTPixelDecoder


class AutoEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is not None:
            self.config = config
            self.encoder = MultiModalityCausalLM.from_pretrained(config.janus_path, trust_remote_code=True).vision_model
            self.decoder = ViTPixelDecoder(config.decoder)
        else:
            self.encoder = None
            self.decoder = None

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
    def forward_with_feature_dim_down(self, x):
        z = self.encoder(x)
        z = self.decoder.get_feature_dim_down(z)
        return self.decoder.decode_feature_dim_down(z)