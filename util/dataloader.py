import os
import torch
import glob
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as pth_transforms
import PIL
from PIL import Image
import sys
import warnings

# 忽略编码警告
warnings.filterwarnings('ignore', category=UnicodeWarning)

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# 设置默认编码为UTF-8
if sys.platform.startswith('linux'):
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        except:
            pass

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

def safe_convert_label(label):
    """安全地转换标签"""
    try:
        if isinstance(label, str):
            # 处理可能的编码问题
            try:
                return int(label)
            except (ValueError, UnicodeDecodeError):
                # 尝试使用不同的编码
                try:
                    label_bytes = label.encode('latin-1', errors='replace')
                    label_str = label_bytes.decode('utf-8', errors='replace')
                    return int(label_str)
                except:
                    return 0  # 默认标签
        elif isinstance(label, (int, float)):
            return int(label)
        else:
            return int(label)
    except Exception as e:
        print(f"标签转换错误: {e}, 使用默认标签0")
        return 0

def collate_fn_imagenet_wds(batch):
    pixel_values = []
    labels = []
    
    for sample in batch:
        try:
            # 安全地处理图像
            img = safe_transform(sample["jpg"])
            if img is None:
                continue
                
            # 安全地处理标签
            label = safe_convert_label(sample["cls"])
            labels.append(label)
            pixel_values.append(img)
            
        except (UnicodeDecodeError, UnicodeEncodeError) as e:
            print(f"Unicode编码错误，跳过样本: {e}")
            continue
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

        # 设置tarfile编码
        import tarfile
        tarfile.ENCODING = 'utf-8'

        # 添加错误处理和编码设置
        try:
            imagenet_wds_train = load_dataset(
                "webdataset",
                data_files = data_files,
                split      = "train",
                streaming   = True,
            )
        except Exception as e:
            print(f"数据集加载错误: {e}")
            # 尝试使用不同的编码设置
            tarfile.ENCODING = 'latin-1'
            imagenet_wds_train = load_dataset(
                "webdataset",
                data_files = data_files,
                split      = "train",
                streaming   = True,
            )
        
        # 使用单进程以避免编码问题
        num_workers = 0
        
        dataloader = DataLoader(
            imagenet_wds_train,
            batch_size  = config.batch_size,
            collate_fn  = collate_fn_imagenet_wds,
            num_workers = num_workers,
            drop_last   = True,
            persistent_workers = False,  # 禁用持久化工作进程
        )
        if getattr(config, "val_path", None) is None:
            return dataloader
        else:
            raise NotImplementedError
