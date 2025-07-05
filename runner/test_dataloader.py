from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
import glob
import torch
from PIL import Image
from torchvision import transforms as pth_transforms

accelerator = Accelerator()
process_index = accelerator.process_index
num_processes = accelerator.num_processes

data_path = "/data/phd/jinjiachun/dataset/timm/imagenet-1k-wds"
# data_path = "/data1/LargeData/BLIP3o-Pretrain-JourneyDB"
all_data_files = glob.glob(os.path.join(data_path, "*.tar"))

imagenet_transform_train = pth_transforms.Compose([
    pth_transforms.Resize(384, max_size=None),
    pth_transforms.RandomHorizontalFlip(p=0.5),
    pth_transforms.CenterCrop(384),
    pth_transforms.ToTensor(),
    pth_transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # 将单通道转换为三通道
])

def safe_load_image(image_data):
    try:
        if isinstance(image_data, Image.Image):
            return image_data
        
        if hasattr(image_data, "read"):
            image_data.seek(0)
            img = Image.open(image_data)
            img.load()
            return img
        else:
            img = Image.open(image_data)
            img.load()
            return img
    except (UnicodeDecodeError, OSError, ValueError, Exception) as e:
        print(f"Error in safe_load_image(): {e}")
        return None

def collate_fn_mine(batch):
    pixel_values = []
    try:
        for sample in batch:
            try:
                img = safe_load_image(sample["jpg"])
                if img is None:
                    continue
                
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                width, height = img.width, img.height
                if max(width, height) < 384:
                    continue
                
                pixel_value = imagenet_transform_train(img)
                pixel_values.append(pixel_value)
            except Exception as e:
                print(f"Error in collate_fn_mine(): {e}")
                continue
    except Exception as e:
        print(f"Error in collate_fn_mine(): {e}")

    if len(pixel_values) == 0:
        print(f"All images were skipped")
        return {"pixel_values": torch.empty(0, 3, 384, 384)}

    pixel_values = torch.stack(pixel_values, dim=0)
    return {"pixel_values": pixel_values}

train_dataset = load_dataset(
    "webdataset",
    data_files = all_data_files,
    split      = "train",
    streaming  = True,
)

# 针对 HuggingFace Streaming dataset，使用 .shard 切片
sharded_train_dataset = train_dataset.shard(num_shards=num_processes, index=process_index)

# DataLoader设置
train_dataloader = DataLoader(
    sharded_train_dataset,
    batch_size  = 100,
    num_workers = 8,
    collate_fn  = collate_fn_mine,
)

train_dataloader = accelerator.prepare(train_dataloader)

iters = 0
n = 0
for batch in train_dataloader:
    iters += 1
    n += batch["pixel_values"].shape[0]

    if iters % 100 == 0:
        print(accelerator.process_index, iters, n)


print("finished", accelerator.process_index, iters, n)