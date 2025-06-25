from diffusers import DDPMScheduler, DDIMScheduler
from .dit import DiTClassConditional
from .dit_fb import DiT_fb

def get_dit(config):
    if config.type == "dit_class_conditional":
        dit = DiTClassConditional(config)
    elif config.type == "dit_fb":
        dit = DiT_fb(config)
    elif config.type is None:
        dit = None
    
    return dit

def get_diffusion_scheduler():
    train_scheduler = DDPMScheduler(
        beta_schedule          = "scaled_linear",
        beta_start             = 0.00085,
        beta_end               = 0.012,
        num_train_timesteps    = 1000,
        clip_sample            = False,
        prediction_type        = "v_prediction",
        # set_alpha_to_one       = True,
        steps_offset           = 1,
        trained_betas          = None,
        timestep_spacing       = "trailing",
        rescale_betas_zero_snr = True
    )

    sample_scheduler = DDIMScheduler(
        beta_schedule          = "scaled_linear",
        beta_start             = 0.00085,
        beta_end               = 0.012,
        num_train_timesteps    = 1000,
        clip_sample            = False,
        prediction_type        = "v_prediction",
        set_alpha_to_one       = True,
        steps_offset           = 1,
        trained_betas          = None,
        timestep_spacing       = "trailing",
        rescale_betas_zero_snr = True
    )

    return train_scheduler, sample_scheduler