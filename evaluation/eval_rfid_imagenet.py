import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchvision.transforms as pth_transforms
from omegaconf import OmegaConf
from accelerate import Accelerator
from util.dataloader import get_imagenet_wds_val_dataloader
from evaluation.eval_rfid_imagenet_basic import AutoEncoder

exp_dir = "/data/phd/jinjiachun/experiment/decoder/0619_decoder"
config_path = os.path.join(exp_dir, "config.yaml")
config = OmegaConf.load(config_path)
config.data.batch_size = 1

dataloader = get_imagenet_wds_val_dataloader(config.data)

autoencoder = AutoEncoder(config)

autoencoder.decoder.load_state_dict(torch.load(os.path.join(exp_dir, "Decoder-decoder-425k"), map_location="cpu", weights_only=True), strict=True)
autoencoder.eval()

accelerator = Accelerator()
autoencoder, dataloader = accelerator.prepare(autoencoder, dataloader)
rank = accelerator.state.local_process_index
world_size = accelerator.state.num_processes

with torch.no_grad():
    for batch in dataloader:
        x = batch["pixel_values"]
        print(x.shape, x.device)
        break