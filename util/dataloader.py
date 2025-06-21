import torch
import glob
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as pth_transforms
import os
import warnings
import io
from PIL import Image
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置PIL忽略EXIF错误
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

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
                        continue
                elif isinstance(jpg_data, Image.Image):
                    img = jpg_data
                else:
                    print(f"Unexpected image type: {type(jpg_data)}, skipping")
                    continue
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

            img = safe_image_transform(img)
            if img is None:
                continue
            if img.shape[0] != 3:
                print("skip", img.shape)
                continue
            labels.append(sample["cls"])
            pixel_values.append(img)
    except Exception as e:
        print(f"处理样本时出错: {e}")
    # 如果所有图像都被跳过，返回空批次
    if len(pixel_values) == 0:
        return {"pixel_values": torch.empty(0, 3, 384, 384), "labels": torch.empty(0)}
    
    pixel_values = torch.stack(pixel_values, dim=0)
    labels = torch.Tensor(labels)
    
    return {"pixel_values": pixel_values, "labels": labels}

class RobustDataLoader:
    """具有错误处理的数据加载器包装器"""
    
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = None
        self.skipped_batches = 0
        self.total_batches = 0
        self._reset_iterator()
    
    def _reset_iterator(self):
        """重置数据加载器迭代器"""
        try:
            self.iterator = iter(self.dataloader)
        except Exception as e:
            print(f"重置数据加载器迭代器时出错: {e}")
            self.iterator = None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """获取下一个批次，具有错误处理"""
        while True:
            try:
                if self.iterator is None:
                    raise RuntimeError("无法创建数据加载器迭代器")
                
                batch = next(self.iterator)
                self.total_batches += 1
                return batch
                
            except (UnicodeDecodeError, OSError, ValueError, RuntimeError) as e:
                print(f"数据加载错误: {e}")
                self.skipped_batches += 1
                print(f"跳过损坏的批次 (总计跳过: {self.skipped_batches})")
                
                # 尝试获取下一个批次
                try:
                    batch = next(self.iterator)
                    self.total_batches += 1
                    return batch
                except StopIteration:
                    # 如果已经到达数据集末尾，正常停止
                    raise StopIteration
            
            except StopIteration:
                # 数据加载器已经遍历完毕，正常停止
                raise StopIteration
    
    def get_stats(self):
        """获取数据加载统计信息"""
        return {
            "total_batches": self.total_batches,
            "skipped_batches": self.skipped_batches,
            "skip_rate": self.skipped_batches / max(self.total_batches, 1) * 100
        }

def get_dataloader(config):
    if config.name == "imagenet_wds":
        data_files = glob.glob(os.path.join(config.train_path, "*.tar"))
        
        if not data_files:
            raise ValueError(f"在路径 {config.train_path} 中没有找到 .tar 文件")

        imagenet_wds_train = load_dataset(
            "webdataset",
            data_files = data_files,
            split      = "train",
            num_proc   = 8,
            streaming  = False,  # 确保不使用流式加载以避免EXIF错误
        )

        dataloader = DataLoader(
            imagenet_wds_train,
            batch_size  = config.batch_size,
            collate_fn  = collate_fn_imagenet_wds,
            shuffle     = True,
            num_workers = config.num_workers,
            drop_last   = True,
            persistent_workers = True if config.num_workers > 0 else False,  # 保持worker进程以避免重复初始化
        )
        
        # 包装数据加载器以添加错误处理
        robust_dataloader = RobustDataLoader(dataloader)
        
        if getattr(config, "val_path", None) is None:
            return robust_dataloader
        else:
            raise NotImplementedError

def get_imagenet_wds_val_dataloader(config):
    if config.name == "imagenet_wds":
        data_files = glob.glob(os.path.join(config.train_path, "*validation*.tar"))

        imagenet_wds_val = load_dataset(
            "webdataset",
            data_files = data_files,
            split      = "validation",
            num_proc   = 8,
            streaming  = False,  # 确保不使用流式加载以避免EXIF错误
        )

        dataloader = DataLoader(
            imagenet_wds_val,
            batch_size  = config.batch_size,
            collate_fn  = collate_fn_imagenet_wds,
            shuffle     = False,
            num_workers = config.num_workers,
            drop_last   = False,
            persistent_workers = True if config.num_workers > 0 else False,  # 保持worker进程以避免重复初始化
        )

        return dataloader