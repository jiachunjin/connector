import os
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf
import torch
from tqdm import tqdm

from model.decoder import get_decoder
from model.loss.rec_loss import RecLoss
from util.misc import process_path_for_different_machine, flatten_dict
from util.dataloader import get_dataloader
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

def main():
    config = OmegaConf.load("config/learn_to_use.yaml")
    config = process_path_for_different_machine(config)
    accelerator, output_dir = get_accelerator(config.train)
    accelerator.print(config)

    # Load models and dataloader
    decoder = get_decoder(config.decoder)
    janus = MultiModalityCausalLM.from_pretrained(config.janus_path, trust_remote_code=True)
    extractor = janus.vision_model
    rec_loss = RecLoss(config.rec_loss)
    dataloader = get_dataloader(config.data)

    if config.train.resume_path_decoder is not None:
        ckpt = torch.load(config.train.resume_path_decoder, map_location="cpu")
        if config.train.skipped_keys:
            ckpt = {k: v for k, v in ckpt.items() if k not in config.train.skipped_keys}
        m, u = decoder.load_state_dict(ckpt, strict=False)
        accelerator.print(f"Loaded {m} modules and {u} unmatch modules")
    if config.train.resume_path_recloss is not None:
        ckpt = torch.load(config.train.resume_path_recloss, map_location="cpu")
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

    decoder, rec_loss, dataloader, optimizer, optimizer_disc = accelerator.prepare(decoder, rec_loss, dataloader, optimizer, optimizer_disc)
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

    # while not training_done:
    for batch in dataloader:
        with accelerator.accumulate([decoder, rec_loss]):
            decoder.train()
            rec_loss.train()
            x = batch["pixel_values"]
            x = x.to(dtype)
            x = x * 2 - 1

            with torch.no_grad():
                feature = extractor(x).to(dtype)
            rec = decoder(feature)

            print(rec.shape, x.shape)
            break

if __name__ == "__main__":
    main()