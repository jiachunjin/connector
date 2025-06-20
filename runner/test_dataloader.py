import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from util.dataloader import get_dataloader
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import os
from util.misc import process_path_for_different_machine
from tqdm import tqdm

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

    dataloader = get_dataloader(config.data)

    dataloader = accelerator.prepare(dataloader)

    global_step = 0
    progress_bar = tqdm(
        total   = config.train.num_iter,
        initial = 0,
        desc    = "Steps",
        disable = not accelerator.is_local_main_process,
    )
    for batch in dataloader:
        if batch["pixel_values"].shape[0] == 0:
            print("跳过空批次")
            continue
        x = batch["pixel_values"]
        if accelerator.sync_gradients:
            global_step += 1
            progress_bar.update(1)
    print("test passed")
    # for batch in dataloader:
    #     print(batch.keys())
    #     break

if __name__ == "__main__":
    main()