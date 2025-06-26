import torch
from tqdm import tqdm

@torch.no_grad()
def dit_generate(dit, sample_scheduler, decoder, y, cfg_scale):
    sample_scheduler.set_timesteps(50)
    B = y.shape[0]

    pred_latents = torch.randn((B, 576, 32), device=y.device)
    pred_latents *= sample_scheduler.init_noise_sigma

    for t in tqdm(sample_scheduler.timesteps):
        pred_latents = sample_scheduler.scale_model_input(pred_latents, t)
        t_sample = torch.as_tensor([t], device=y.device)
        
        y_uncond = torch.full((B,), 1000, device=y.device, dtype=y.dtype)
        y_batch = torch.cat([y_uncond, y], dim=0)
        pred_latents_batch = torch.cat([pred_latents, pred_latents], dim=0)
        noise_pred = dit(pred_latents_batch, t_sample.repeat(2*B), y_batch)
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        pred_latents = sample_scheduler.step(noise_pred, t, pred_latents).prev_sample
    
    output = decoder.decode_feature_dim_down(pred_latents)
    
    return output



if __name__ == "__main__":
    import torchvision.utils as vutils
    import torchvision.transforms as pth_transforms
    from omegaconf import OmegaConf
    from model.dit import get_dit, get_diffusion_scheduler
    from model.decoder import get_decoder
    

    config_path = "/data/phd/jinjiachun/experiment/dit/0625_dit_siglip_32_class_conditional/config.yaml"
    ckpt_path = "/data/phd/jinjiachun/experiment/dit/0625_dit_siglip_32_class_conditional/DiT-dit-25000"
    config = OmegaConf.load(config_path)

    dit = get_dit(config.dit)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    dit.load_state_dict(ckpt, strict=True)
    dit.eval()
    dit.requires_grad_(False)

    decoder = get_decoder(config.decoder)
    ckpt_decoder = torch.load(config.decoder.pretrained_path, map_location="cpu", weights_only=True)
    decoder.load_state_dict(ckpt_decoder, strict=True)
    decoder.eval()
    decoder.requires_grad_(False)

    _, sample_scheduler = get_diffusion_scheduler()

    device = torch.device("cuda:7")
    dit.to(device)
    decoder.to(device)

    y = torch.LongTensor([22, 980]).to(device)
    cfg_scale = 2.0

    output = dit_generate(dit, sample_scheduler, decoder, y, cfg_scale)
    print(output.shape)

    # 将输出从[-1,1]范围转换到[0,1]范围
    output = ((output + 1) / 2).clamp(0, 1)
    
    # 创建网格图像
    grid = vutils.make_grid(output, nrow=2, padding=10, normalize=False)
    
    # 转换为PIL图像并保存
    inversed_transform = pth_transforms.ToPILImage()
    grid_image = inversed_transform(grid)
    grid_image.save("output_grid.png")
    
    print(f"Grid image saved as output_grid.png with shape: {grid.shape}")