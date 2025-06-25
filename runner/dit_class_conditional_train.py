import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pprint
import argparse
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf
from tqdm import tqdm

from model.decoder import get_decoder
from model.dit import get_dit, get_diffusion_scheduler
from util.dataloader import get_dataloader
from util.misc import process_path_for_different_machine, flatten_dict
from janus.models import MultiModalityCausalLM

def get_accelerator(config):
    output_dir = os.path.join(config.root, config.exp_name, config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = os.path.join(output_dir, config.logging_dir)
    project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        log_with                    = None if config.report_to == "no" else config.report_to,
        mixed_precision             = config.mixed_precision,
        project_config              = project_config,
        gradient_accumulation_steps = config.gradient_accumulation_steps,
    )

    return accelerator, output_dir


def main(args):
    config = OmegaConf.load(args.config)
    config = process_path_for_different_machine(config)
    accelerator, output_dir = get_accelerator(config.train)
    accelerator.print("Configuration:")
    accelerator.print(pprint.pformat(OmegaConf.to_container(config, resolve=True), indent=2, width=120).strip('{}'))

    # Load models and dataloader
    decoder = get_decoder(config.decoder)
    if config.decoder.pretrained_path is not None:
        decoder.load_state_dict(torch.load(config.decoder.pretrained_path, map_location="cpu", weights_only=True), strict=True)
        accelerator.print(f"Loaded decoder from {config.decoder.pretrained_path}")
    else:
        accelerator.print("Warning: No pretrained path provided !!!")
    decoder.eval()
    decoder.requires_grad_(False)
    janus = MultiModalityCausalLM.from_pretrained(config.janus_path, trust_remote_code=True)
    extractor = janus.vision_model

    dit = get_dit(config.dit)
    train_scheduler, _ = get_diffusion_scheduler()
    dataloader = get_dataloader(config.data)

    if config.train.resume_path is not None:
        ckpt = torch.load(config.train.resume_path, map_location="cpu", weights_only=True)
        if config.train.skipped_keys:
            ckpt = {k: v for k, v in ckpt.items() if k not in config.train.skipped_keys}
        m, u = dit.load_state_dict(ckpt, strict=False)
        accelerator.print(f"Loaded {m} modules and {u} unmatch modules")

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(dit.parameters())

    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = config.train.lr,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )

    if accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    dit, dataloader, optimizer = accelerator.prepare(dit, dataloader, optimizer)
    dit = dit.to(dtype)
    decoder = decoder.to(accelerator.device, dtype)
    extractor = extractor.to(accelerator.device, dtype)

    config.device_count = accelerator.num_processes
    if accelerator.is_main_process:
        accelerator.init_trackers(config.train.wandb_proj, config=flatten_dict(config))
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config, f)

    training_done = False
    epoch = 0

    progress_bar = tqdm(
        total   = config.train.num_iter,
        initial = global_step,
        desc    = "Steps",
        disable = not accelerator.is_local_main_process,
    )

    if accelerator.is_main_process:
        print(f"Learnable parameters: {sum(p.numel() for p in params_to_learn if p.requires_grad) / 1e6} M")

    if accelerator.is_main_process:
        print(f"dit dtype: {next(dit.parameters()).dtype}")
        print(f"Accelerator mixed precision: {accelerator.mixed_precision}")

    while not training_done:
        for batch in dataloader:
            if batch["pixel_values"].shape[0] == 0:
                print("跳过空批次")
                continue

            with accelerator.accumulate(dit):
                dit.train()
            
                x = batch["pixel_values"].to(dtype)
                x = x * 2 - 1
                y = batch["labels"]
                with torch.no_grad():
                    feature = extractor(x).to(dtype)
                    x_0 = decoder.get_feature_dim_down(feature)
                    x_0 *= config.dit.scale_factor

                # Diffusion training
                B = x_0.shape[0]
                timesteps = torch.randint(0, 1000, (B,), dtype=torch.int64, device=accelerator.device)
                noise = torch.randn_like(x_0, device=accelerator.device, dtype=x_0.dtype)
                x_t = train_scheduler.add_noise(x_0, noise, timesteps)
                target = train_scheduler.get_velocity(x_0, noise, timesteps)
                pred = dit(x_t, y, timesteps).to(dtype)
                loss = torch.nn.functional.mse_loss(pred, target, reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    optimizer.zero_grad()
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                    optimizer.step()

                global_step += 1
                progress_bar.update(1)

                logs = dict(
                    dit_loss = accelerator.gather(loss.detach()).mean().item(),
                )
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)

            if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                dit.eval()
                state_dict = accelerator.unwrap_model(dit).state_dict()
                save_path = os.path.join(output_dir, f"DiT-{config.train.exp_name}-{global_step}")
                torch.save(state_dict, save_path)
                print(f"DiT model saved to {save_path}")

            accelerator.wait_for_everyone()

            if global_step >= config.train.num_iter:
                training_done = True
                break

        epoch += 1
        accelerator.print(f"epoch {epoch}: finished")
        accelerator.log({"epoch": epoch}, step=global_step)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)