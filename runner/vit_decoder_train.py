import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf
import torch
import pprint
from tqdm import tqdm

from model.decoder import get_decoder
from model.loss.rec_loss import RecLoss
from util.misc import process_path_for_different_machine, flatten_dict
from util.dataloader import get_dataloader, get_imagenet_wds_val_dataloader
from util.ema import EMA
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
    janus = MultiModalityCausalLM.from_pretrained(config.janus_path, trust_remote_code=True)
    extractor = janus.vision_model
    rec_loss = RecLoss(config.rec_loss)
    if config.train.resume_path_decoder is not None:
        ckpt = torch.load(config.train.resume_path_decoder, map_location="cpu", weights_only=True)
        if config.train.skipped_keys:
            ckpt = {k: v for k, v in ckpt.items() if k not in config.train.skipped_keys}
        m, u = decoder.load_state_dict(ckpt, strict=False)
        accelerator.print(f"Missed {m} modules and {u} unmatch modules")
    if config.train.resume_path_recloss is not None:
        ckpt = torch.load(config.train.resume_path_recloss, map_location="cpu", weights_only=True)
        rec_loss.load_state_dict(ckpt, strict=True)
    
    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(decoder.parameters())
    disc_params = list(rec_loss.parameters())

    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = config.train.lr,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )

    optimizer_disc = torch.optim.AdamW(
        disc_params,
        lr           = config.train.lr_disc,
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

    dataloader = get_dataloader(config.data)
    dataloader_val = get_imagenet_wds_val_dataloader(config.data)

    decoder, rec_loss, dataloader, dataloader_val, optimizer, optimizer_disc = accelerator.prepare(decoder, rec_loss, dataloader, dataloader_val, optimizer, optimizer_disc)
    extractor = extractor.to(accelerator.device, dtype).eval()
    decoder = decoder.to(dtype)
    rec_loss = rec_loss.to(dtype)

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
        print(f"decoder dtype: {next(decoder.parameters()).dtype}")
        print(f"rec loss dtype: {next(rec_loss.parameters()).dtype}")
        print(f"extractor dtype: {next(extractor.parameters()).dtype}")
        print(f"Accelerator mixed precision: {accelerator.mixed_precision}")

    if accelerator.is_main_process:
        ema = EMA(decoder.module, decay=0.999)

    while not training_done:
        for batch in dataloader:
            # 检查批次是否为空
            if batch["pixel_values"].shape[0] == 0:
                print("跳过空批次")
                continue
                
            with accelerator.accumulate([decoder, rec_loss]):
                decoder.train()
                rec_loss.train()
                x = batch["pixel_values"]
                x = x.to(device=accelerator.device, dtype=dtype)
                x = x * 2 - 1

                with torch.no_grad():
                    feature = extractor(x).to(dtype)

                rec = decoder(feature, x)
                # ---------- train autoencoder ----------
                loss_rec, loss_rec_dict = rec_loss(x, rec, global_step, "generator")

                optimizer.zero_grad()
                accelerator.backward(loss_rec)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                optimizer.step()

                if accelerator.is_main_process:
                    ema.update(decoder.module)
                # ---------- train discriminator ----------
                loss_disc, loss_disc_dict = rec_loss(x, rec, global_step, "discriminator")

                optimizer_disc.zero_grad()
                accelerator.backward(loss_disc)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(disc_params, 1.0)
                optimizer_disc.step()

                if accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)

                    logs = dict(
                        loss_rec  = accelerator.gather(loss_rec.detach()).mean().item(),
                        loss_disc = accelerator.gather(loss_disc.detach()).mean().item(),
                        **loss_rec_dict,
                        **loss_disc_dict,
                    )
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

            if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                decoder.eval()
                state_dict = accelerator.unwrap_model(decoder).state_dict()
                torch.save(state_dict, os.path.join(output_dir, f"Decoder-{config.train.exp_name}-{global_step // 1000}k"))

                state_dict = accelerator.unwrap_model(rec_loss).state_dict()
                torch.save(state_dict, os.path.join(output_dir, f"Loss-{config.train.exp_name}-{global_step // 1000}k"))

                ema_path = os.path.join(output_dir, f"EMA-{config.train.exp_name}-{global_step // 1000}k")
                ema.save_shadow(ema_path)
            
            if global_step > 0 and global_step % config.train.val_every == 0:
                decoder.eval()
                import torch_fidelity
                import torchvision.transforms as pth_transforms
                from evaluation.eval_rfid_imagenet_basic import AutoEncoder
                autoencoder = AutoEncoder()
                autoencoder.encoder = extractor
                autoencoder.decoder = decoder.module
                autoencoder.to(accelerator.device)
                autoencoder.eval()
                autoencoder.to(torch.float32)

                rank = accelerator.state.local_process_index
                world_size = accelerator.state.num_processes

                try:
                    os.makedirs(os.path.join(output_dir, "rec_img"), exist_ok=True)
                except Exception as e:
                    print(f"创建evaluation目录失败: {e}")
                    continue

                try:
                    with torch.no_grad():
                        for i, batch in tqdm(enumerate(dataloader_val)):
                            x = batch["pixel_values"]
                            x = x.to(device=accelerator.device, dtype=torch.float32)
                            x = x * 2 - 1
                            rec = autoencoder.forward_with_feature_dim_down(x)

                            x = ((x + 1) / 2).clamp(0, 1)
                            rec = ((rec + 1) / 2).clamp(0, 1)

                            rec = pth_transforms.ToPILImage()(rec.cpu().squeeze(0))
                            rec.save(f"{output_dir}/rec_img/{rank}_{i}.png")

                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        metrics_dict = torch_fidelity.calculate_metrics(
                            input1  = "evaluation/ori_img",
                            input2  = f"{output_dir}/rec_img",
                            cuda    = True,
                            isc     = True,
                            fid     = True,
                            kid     = True,
                            prc     = True,
                            verbose = True,
                        )
                        print(metrics_dict)
                        accelerator.log(metrics_dict, step=global_step)
                except Exception as e:
                    print(f"评估过程中出现错误: {e}")
                    print(f"global_step: {global_step}")
                autoencoder.to(dtype)
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
    parser.add_argument("--config", type=str, default="config/bi_tok.yaml")
    args = parser.parse_args()
    main(args)