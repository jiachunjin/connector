import os
import glob
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as pth_transforms

from datasets import load_dataset
from PIL import Image
from io import BytesIO
from torchvision import transforms
import torch
from tqdm import tqdm
import warnings

# 设置PIL忽略EXIF错误
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

imagenet_transform_train = pth_transforms.Compose([
    pth_transforms.Resize(384, max_size=None),
    pth_transforms.RandomHorizontalFlip(p=0.5),
    pth_transforms.CenterCrop(384),
    pth_transforms.ToTensor(),
    pth_transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # 将单通道转换为三通道
])


def safe_load_image(image_data):
    """安全地加载图像，处理EXIF错误"""
    try:
        # 如果是PIL图像，直接返回
        if isinstance(image_data, Image.Image):
            return image_data
        
        # 如果是字节数据，尝试加载
        if hasattr(image_data, "read"):
            # 重置到开始位置
            image_data.seek(0)
            img = Image.open(image_data)
            # 立即加载图像数据以避免延迟加载时的EXIF错误
            img.load()
            return img
        else:
            # 其他情况，尝试直接打开
            img = Image.open(image_data)
            img.load()
            return img
    except (UnicodeDecodeError, OSError, ValueError, Exception) as e:
        print(f"加载图像时出错: {e}")
        return None


def main():
    data_path = "/data/phd/jinjiachun/dataset/timm/imagenet-1k-wds"
    # data_path = "/data1/LargeData/BLIP3o-Pretrain-JourneyDB"
    accelerator = Accelerator()
    all_data_files = glob.glob(os.path.join(data_path, "*.tar"))
    process_data_files = all_data_files[accelerator.process_index::accelerator.num_processes]

    imagenet_transform_train = pth_transforms.Compose([
        pth_transforms.Resize(384, max_size=None),
        pth_transforms.RandomHorizontalFlip(p=0.5),
        pth_transforms.CenterCrop(384),
        pth_transforms.ToTensor(),
        pth_transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # 将单通道转换为三通道
    ])

    train_dataset = load_dataset(
        "webdataset",
        data_files = process_data_files,
        split      = "train",
        streaming  = True,
    )

    def collate_fn_mine(batch):
        pixel_values = []
        skipped_count = 0
        try:
            for sample in batch:
                try:
                    # 安全地加载图像
                    img = safe_load_image(sample["jpg"])
                    if img is None:
                        skipped_count += 1
                        continue
                    
                    # 转换为RGB模式
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    
                    width, height = img.width, img.height
                    if max(width, height) < 384:
                        skipped_count += 1
                        continue
                    
                    pixel_value = imagenet_transform_train(img)
                    pixel_values.append(pixel_value)
                except Exception as e:
                    print(f"Error in collate_fn_mine(): {e}")
                    skipped_count += 1
                    continue
        except Exception as e:
            print(f"处理样本时出错, collate_fn_mine: {e}")

        if len(pixel_values) == 0:
            print(f"批次中所有图像都被跳过 (跳过数量: {skipped_count})")
            return {"pixel_values": torch.empty(0, 3, 384, 384)}

        pixel_values = torch.stack(pixel_values, dim=0)
        return {"pixel_values": pixel_values}

    dataloader = DataLoader(train_dataset, batch_size=200, collate_fn=collate_fn_mine, num_workers=8)

    num_samples = 0
    iters = 0
    for batch in dataloader:
        iters += 1
        num_samples += batch["pixel_values"].shape[0]
        if iters % 10 == 0:
            print(accelerator.process_index, iters, num_samples)
    print(accelerator.process_index, num_samples)

    accelerator.end_training() # 释放资源

if __name__ == "__main__":
    main()