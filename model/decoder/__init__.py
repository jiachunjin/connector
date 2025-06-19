from .vit_pixel_decoder import ViTPixelDecoder

def get_decoder(config):
    if config.type == "vit_pixel_decoder":
        decoder = ViTPixelDecoder(config)
    elif config.type is None:
        decoder = None
    
    return decoder