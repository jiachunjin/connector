import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch_fidelity
import torchvision.transforms as pth_transforms
from omegaconf import OmegaConf
from accelerate import Accelerator
from util.dataloader import get_imagenet_wds_val_dataloader
from evaluation.eval_rfid_imagenet_basic import AutoEncoder
from tqdm import tqdm

exp_dir = "/data/phd/jinjiachun/experiment/decoder/0619_decoder"
config_path = os.path.join(exp_dir, "config.yaml")
config = OmegaConf.load(config_path)
config.data.batch_size = 1

dataloader = get_imagenet_wds_val_dataloader(config.data)

autoencoder = AutoEncoder(config)

autoencoder.decoder.load_state_dict(torch.load(os.path.join(exp_dir, "Decoder-decoder-485k"), map_location="cpu", weights_only=True), strict=True)
autoencoder.eval()

accelerator = Accelerator()
autoencoder, dataloader = accelerator.prepare(autoencoder, dataloader)
rank = accelerator.state.local_process_index
world_size = accelerator.state.num_processes

os.makedirs("evaluation/rec_img", exist_ok=True)
os.makedirs("evaluation/ori_img", exist_ok=True)

with torch.no_grad():
    for i, batch in tqdm(enumerate(dataloader)):
        x = batch["pixel_values"]
        x = x * 2 - 1
        rec = autoencoder(x)

        x = ((x + 1) / 2).clamp(0, 1)
        rec = ((rec + 1) / 2).clamp(0, 1)

        rec = pth_transforms.ToPILImage()(rec.cpu().squeeze(0))
        ori = pth_transforms.ToPILImage()(x.cpu().squeeze(0))

        rec.save(f"evaluation/rec_img/{rank}_{i}.png")
        ori.save(f"evaluation/ori_img/{rank}_{i}.png")

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    metrics_dict = torch_fidelity.calculate_metrics(
        input1  = "evaluation/ori_img",
        input2  = "evaluation/rec_img",
        cuda    = True,
        isc     = True,
        fid     = True,
        kid     = True,
        prc     = True,
        verbose = True,
    )
    print(metrics_dict)