import torch
import glob
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as pth_transforms
import os
import warnings
import io
from PIL import Image
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置PIL忽略EXIF错误
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

class SafeImageDataset(Dataset):
    """安全的数据集包装器，预处理图像以避免EXIF错误"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.valid_indices = []
        self._filter_valid_samples()
    
    def _filter_valid_samples(self):
        """过滤出有效的样本索引"""
        print("正在预处理数据集，过滤有问题的图像...")
        for i in range(len(self.dataset)):
            try:
                sample = self.dataset[i]
                # 尝试安全地加载图像
                if self._is_valid_sample(sample):
                    self.valid_indices.append(i)
            except Exception as e:
                print(f"跳过样本 {i}: {e}")
                continue
        
        print(f"数据集预处理完成: {len(self.valid_indices)}/{len(self.dataset)} 个有效样本")
    
    def _is_valid_sample(self, sample):
        """检查样本是否有效"""
        try:
            jpg_data = sample["jpg"]
            if isinstance(jpg_data, bytes):
                # 尝试加载图像但不保留，只验证是否有效
                img = Image.open(io.BytesIO(jpg_data))
                img.load()
                img.close()
            elif isinstance(jpg_data, Image.Image):
                # 如果是PIL图像，验证是否有效
                img = jpg_data
                if hasattr(img, 'getexif'):
                    try:
                        img.getexif()
                    except UnicodeDecodeError:
                        return False
            else:
                return False
            
            return True
        except (UnicodeDecodeError, OSError, ValueError, Exception):
            return False
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        return self.dataset[original_idx]

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

imagenet_transform_train = pth_transforms.Compose([
    pth_transforms.Resize(384, max_size=None),
    pth_transforms.RandomHorizontalFlip(p=0.5),
    pth_transforms.CenterCrop(384),
    pth_transforms.ToTensor(),
    pth_transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # 将单通道转换为三通道
])

def safe_image_transform(img):
    """安全地处理图像，跳过有EXIF错误的图像"""
    try:
        # 首先安全地加载图像
        safe_img = safe_load_image(img)
        if safe_img is None:
            return None
        
        # 转换为RGB模式
        if safe_img.mode != "RGB":
            safe_img = safe_img.convert("RGB")
        
        return imagenet_transform_train(safe_img)
    except (UnicodeDecodeError, OSError, ValueError) as e:
        # 跳过有问题的图像
        print(f"跳过有问题的图像: {e}")
        return None

def collate_fn_imagenet_wds(batch):
    pixel_values = []
    labels = []
    skipped_count = 0
    
    try:
        for sample in batch:
            try:
                jpg_data = sample["jpg"]
                if isinstance(jpg_data, bytes):
                    try:
                        img = Image.open(io.BytesIO(jpg_data))
                        img.load()  # Force load to validate
                    except Exception as e:
                        print(f"Skipping corrupted image: {e}")
                        skipped_count += 1
                        continue
                elif isinstance(jpg_data, Image.Image):
                    img = jpg_data
                else:
                    print(f"Unexpected image type: {type(jpg_data)}, skipping")
                    skipped_count += 1
                    continue
            except Exception as e:
                print(f"Error processing sample: {e}")
                skipped_count += 1
                continue

            img = safe_image_transform(img)
            if img is None:
                skipped_count += 1
                continue
            if img.shape[0] != 3:
                print("skip", img.shape)
                skipped_count += 1
                continue
            labels.append(sample["cls"])
            pixel_values.append(img)
    except Exception as e:
        print(f"处理样本时出错: {e}")
        skipped_count += 1
    
    # 如果所有图像都被跳过，返回空批次
    if len(pixel_values) == 0:
        print(f"批次中所有图像都被跳过 (跳过数量: {skipped_count})")
        return {"pixel_values": torch.empty(0, 3, 384, 384), "labels": torch.empty(0)}
    
    pixel_values = torch.stack(pixel_values, dim=0)
    labels = torch.Tensor(labels)
    
    return {"pixel_values": pixel_values, "labels": labels}

def get_dataloader(config):
    if config.name == "imagenet_wds":
        data_files = glob.glob(os.path.join(config.train_path, "*train*.tar"))
        
        if not data_files:
            raise ValueError(f"在路径 {config.train_path} 中没有找到 .tar 文件")

        imagenet_wds_train = load_dataset(
            "webdataset",
            data_files = data_files,
            split      = "train",
            num_proc   = 8,
            streaming  = False,  # 确保不使用流式加载以避免EXIF错误
        )

        # 使用安全的数据集包装器
        safe_dataset = SafeImageDataset(imagenet_wds_train)

        dataloader = DataLoader(
            safe_dataset,
            batch_size  = config.batch_size,
            collate_fn  = collate_fn_imagenet_wds,
            shuffle     = True,
            num_workers = config.num_workers,
            drop_last   = True,
            persistent_workers = True if config.num_workers > 0 else False,  # 保持worker进程以避免重复初始化
        )
        
        return dataloader

def get_imagenet_wds_val_dataloader(config):
    if config.name == "imagenet_wds":
        data_files = glob.glob(os.path.join(config.train_path, "*validation*.tar"))

        imagenet_wds_val = load_dataset(
            "webdataset",
            data_files = data_files,
            split      = "train",
            num_proc   = 8,
            streaming  = False,  # 确保不使用流式加载以避免EXIF错误
        )

        # 使用安全的数据集包装器
        safe_dataset = SafeImageDataset(imagenet_wds_val)

        dataloader = DataLoader(
            safe_dataset,
            batch_size  = config.batch_size,
            collate_fn  = collate_fn_imagenet_wds,
            shuffle     = False,
            num_workers = config.num_workers,
            drop_last   = False,
            persistent_workers = True if config.num_workers > 0 else False,  # 保持worker进程以避免重复初始化
        )

        return dataloader