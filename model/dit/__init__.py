from .dit import DiTClassConditional

def get_dit(config):
    if config.type == "dit_class_conditional":
        dit = DiTClassConditional(config)
    elif config.type is None:
        dit = None
    
    return dit