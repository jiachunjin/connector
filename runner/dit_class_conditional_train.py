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

from model.dit import get_dit
from util.dataloader import get_dataloader
from util.misc import process_path_for_different_machine, flatten_dict

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
    dit = get_dit(config.dit)
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

    num_samples = 0
    while not training_done:
        for batch in dataloader:
            if batch["pixel_values"].shape[0] == 0:
                print("跳过空批次")
                continue
            
            x = batch["pixel_values"].to(dtype)
            y = batch["labels"]
            print(x.shape, y.shape, y)
            exit(0)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

            if global_step >= config.train.num_iter:
                training_done = True
                break
                
        epoch += 1
        accelerator.print(f"epoch {epoch}: finished")
        accelerator.print(f"num_samples in this epoch {num_samples}")
        accelerator.log({"epoch": epoch}, step=global_step)
        num_samples = 0
        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)