import os
import torch
import glob
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as pth_transforms
import PIL
from PIL import Image

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 禁用PIL的EXIF警告
PIL.Image.MAX_IMAGE_PIXELS = None
# 设置PIL不加载EXIF数据
Image.LOAD_TRUNCATED_IMAGES = True

imagenet_transform_train = pth_transforms.Compose([
    pth_transforms.Resize(384, max_size=None),
    pth_transforms.RandomHorizontalFlip(p=0.5),
    pth_transforms.CenterCrop(384),
    pth_transforms.ToTensor(),
    pth_transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # 将单通道转换为三通道
])

def safe_transform(image):
    """安全地转换图像，处理损坏的文件"""
    try:
        # 确保图像是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 应用变换
        img_tensor = imagenet_transform_train(image)
        
        # 检查通道数
        if img_tensor.shape[0] != 3:
            print(f"跳过非3通道图像: {img_tensor.shape}")
            return None
            
        return img_tensor
    except Exception as e:
        print(f"图像处理错误: {e}")
        return None

def collate_fn_imagenet_wds(batch):
    pixel_values = []
    labels = []
    
    for sample in batch:
        try:
            # 安全地处理图像
            img = safe_transform(sample["jpg"])
            if img is None:
                continue
                
            labels.append(sample["cls"])
            pixel_values.append(img)
            
        except Exception as e:
            print(f"跳过损坏的样本: {e}")
            continue
    
    # 检查是否有有效样本
    if len(pixel_values) == 0:
        print("警告: 批次中没有有效样本，返回空批次")
        return {"pixel_values": torch.empty(0, 3, 384, 384), "labels": torch.empty(0)}
    
    pixel_values = torch.stack(pixel_values, dim=0)
    labels = torch.Tensor(labels)
    
    return {"pixel_values": pixel_values, "labels": labels}

def get_dataloader(config):
    if config.name == "imagenet_wds":
        data_files = glob.glob(os.path.join(config.train_path, "*.tar"))

        imagenet_wds_train = load_dataset(
            "webdataset",
            data_files = data_files,
            split      = "train",
            streaming   = True,
        )
        dataloader = DataLoader(
            imagenet_wds_train,
            batch_size  = config.batch_size,
            collate_fn  = collate_fn_imagenet_wds,
            num_workers = config.num_workers,
            drop_last   = True,
        )
        if getattr(config, "val_path", None) is None:
            return dataloader
        else:
            raise NotImplementedError
