import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from util.dataloader import get_dataloader, get_dataloader_test
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
    config = OmegaConf.load("config/vit_decoder_scale_hybird_data.yaml")
    config = process_path_for_different_machine(config)
    accelerator, output_dir = get_accelerator(config.train)

    dataloader = get_dataloader_test(config.data)

    dataloader = accelerator.prepare(dataloader)

    epoch = 0
    global_step = 0
    progress_bar = tqdm(
        total   = config.train.num_iter,
        initial = 0,
        desc    = "Steps",
        disable = not accelerator.is_local_main_process,
    )
    
    num_samples = 0
    while True:
        for batch in dataloader:
            if batch["pixel_values"].shape[0] == 0:
                print("跳过空批次")
                continue
            num_samples += batch["pixel_values"].shape[0]

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
    main()