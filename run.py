import os
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf

from model.decoder import get_decoder
from util.misc import process_machine_path
from janus.models import MultiModalityCausalLM

def get_accelerator(config):
    print(config)
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
    config = process_machine_path(config)
    accelerator, output_dir = get_accelerator(config.train)
    accelerator.print("Hello, World!")
    decoder = get_decoder(config.decoder)
    janus = MultiModalityCausalLM.from_pretrained(config.janus_path, trust_remote_code=True)
    extractor = janus.vision_model
    num_params = sum(p.numel() for p in extractor.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")

if __name__ == "__main__":
    main()