import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf
import torch
import pprint
from tqdm import tqdm
from diffusers import AutoencoderKL

from util.misc import process_path_for_different_machine, flatten_dict
from util.dataloader import get_dataloader, get_imagenet_wds_val_dataloader, get_wds_dataloader
from model.vae_aligner import get_vae_aligner
# from model.loss.rec_loss import RecLoss
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

    vae_aligner = get_vae_aligner(config.vae_aligner)
    params_to_learn = list(vae_aligner.parameters())
    # rec_loss = RecLoss(config.rec_loss)
    # vae_aligner.rec_loss = rec_loss
    # disc_params = list(vae_aligner.rec_loss.parameters())
    vae = AutoencoderKL.from_pretrained(config.vae_path)
    vae.requires_grad_(False)
    siglip = MultiModalityCausalLM.from_pretrained(config.janus_path, trust_remote_code=True).vision_model

    if config.train.resume_path is not None:
        ckpt = torch.load(config.train.resume_path, map_location="cpu", weights_only=True)
        if config.train.skipped_keys:
            ckpt = {k: v for k, v in ckpt.items() if k not in config.train.skipped_keys}
        m, u = vae_aligner.load_state_dict(ckpt, strict=False)
        accelerator.print(f"Missed {m} modules and {u} unmatch modules")

    global_step = config.train.global_step if config.train.global_step is not None else 0

    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = config.train.lr,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )
    # optimizer_disc = torch.optim.AdamW(
    #     disc_params,
    #     lr           = config.train.lr_disc,
    #     betas        = (0.9, 0.95),
    #     weight_decay = 5e-2,
    #     eps          = 1e-8,
    # )

    if accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # dataloader = get_dataloader(config.data)
    # dataloader_val = get_imagenet_wds_val_dataloader(config.data)
    dataloader = get_wds_dataloader(config.data)

    vae_aligner, dataloader, optimizer = accelerator.prepare(vae_aligner, dataloader, optimizer)
    siglip = siglip.to(accelerator.device, dtype).eval()
    vae = vae.to(accelerator.device, dtype).eval()
    vae_aligner = vae_aligner.to(dtype)

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
        print(f"vae_aligner dtype: {next(vae_aligner.parameters()).dtype}")
        print(f"Accelerator mixed precision: {accelerator.mixed_precision}")

    while not training_done:
        for batch in dataloader:
            # 检查批次是否为空
            if batch["pixel_values"].shape[0] == 0:
                print("跳过空批次")
                continue

            with accelerator.accumulate([vae_aligner]):
                vae_aligner.train()
                x = batch["pixel_values"]
                x = x.to(device=accelerator.device, dtype=dtype)
                x = x * 2 - 1

                with torch.no_grad():
                    x_siglip = siglip(x).to(dtype)
                    vae_latent = vae.encode(x).latent_dist.sample()


                rec_latent = vae_aligner(x_siglip).to(dtype)
                # rec = vae.decode(rec_latent).sample

                loss_mse = torch.nn.functional.mse_loss(rec_latent, vae_latent)
                # loss_rec, loss_rec_dict = vae_aligner.rec_loss(x, rec, global_step, "generator")

                optimizer.zero_grad()
                accelerator.backward(loss_mse)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                optimizer.step()

                # ---------- train discriminator ----------
                # loss_disc, loss_disc_dict = vae_aligner.rec_loss(x, rec, global_step, "discriminator")

                # optimizer_disc.zero_grad()
                # accelerator.backward(loss_disc)
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(disc_params, 1.0)
                # optimizer_disc.step()

                if accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)

                    logs = dict(
                        loss_mse  = accelerator.gather(loss_mse.detach()).mean().item(),
                        # loss_rec  = accelerator.gather(loss_rec.detach()).mean().item(),
                        # loss_disc = accelerator.gather(loss_disc.detach()).mean().item(),
                        # **loss_rec_dict,
                        # **loss_disc_dict,
                    )
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

            if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                vae_aligner.eval()
                state_dict = accelerator.unwrap_model(vae_aligner).state_dict()
                torch.save(state_dict, os.path.join(output_dir, f"vae_aligner-{config.train.exp_name}-{global_step // 1000}k"))

        epoch += 1
        accelerator.print(f"epoch {epoch}: finished")
        accelerator.log({"epoch": epoch}, step=global_step)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/vae_aligner/siglip_flux.yaml")
    args = parser.parse_args()
    main(args)