import torch
import glob
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torchvision.transforms as pth_transforms
import os
import warnings
import io
from PIL import Image
from tqdm import trange
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置PIL忽略EXIF错误
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

class SafeImageDataset(Dataset):
    """安全的数据集包装器，动态处理图像以避免EXIF错误"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.valid_indices = set()  # 使用set来记录已验证的索引
        self.invalid_indices = set()  # 使用set来记录无效的索引
        self._is_iterable = hasattr(dataset, '__iter__') and not hasattr(dataset, '__len__')
        self._dataset_iter = None  # 用于 IterableDataset 的迭代器
    
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
        if self._is_iterable:
            # 对于 IterableDataset，返回一个大的数字作为长度
            # 这只是一个占位符，实际长度在迭代时确定
            return 15000000  # 返回一个足够大的数字
        return len(self.dataset)
    
    def __iter__(self):
        """支持迭代，用于 IterableDataset"""
        if self._is_iterable:
            return self._iter_valid_samples()
        else:
            # 对于普通 Dataset，返回一个迭代器
            return iter([self[i] for i in range(len(self))])
    
    def _iter_valid_samples(self):
        """迭代有效样本"""
        # 创建一个新的迭代器，避免共享状态
        dataset_iter = iter(self.dataset)
        while True:
            try:
                sample = next(dataset_iter)
                if self._is_valid_sample(sample):
                    yield sample
                else:
                    print("跳过无效样本")
                    continue
            except StopIteration:
                break
            except Exception as e:
                print(f"处理样本时出错: {e}")
                continue
    
    def __getitem__(self, idx):
        if self._is_iterable:
            # 对于 IterableDataset，直接返回下一个有效样本
            return self._get_next_valid_sample_iterable()
        
        # 如果已经知道这个索引是无效的，跳过
        if idx in self.invalid_indices:
            # 返回下一个有效样本
            return self._get_next_valid_sample(idx)
        
        # 如果已经验证过是有效的，直接返回
        if idx in self.valid_indices:
            return self.dataset[idx]
        
        # 第一次访问这个索引，验证样本
        try:
            sample = self.dataset[idx]
            if self._is_valid_sample(sample):
                self.valid_indices.add(idx)
                return sample
            else:
                self.invalid_indices.add(idx)
                return self._get_next_valid_sample(idx)
        except Exception as e:
            print(f"样本 {idx} 验证失败: {e}")
            self.invalid_indices.add(idx)
            return self._get_next_valid_sample(idx)
    
    def _get_next_valid_sample_iterable(self):
        """获取 IterableDataset 的下一个有效样本"""
        # 确保迭代器已初始化
        if self._dataset_iter is None:
            self._dataset_iter = iter(self.dataset)
        
        while True:
            try:
                sample = next(self._dataset_iter)
                if self._is_valid_sample(sample):
                    return sample
                else:
                    print("跳过无效样本")
                    continue
            except StopIteration:
                # 重新初始化迭代器，静默地重新开始迭代
                print("数据集迭代结束，重新开始迭代")
                self._dataset_iter = iter(self.dataset)
                continue
            except Exception as e:
                print(f"处理样本时出错: {e}")
                continue
    
    def _get_next_valid_sample(self, current_idx):
        """获取下一个有效样本"""
        # 向前查找下一个有效样本
        for i in range(current_idx + 1, len(self.dataset)):
            if i not in self.invalid_indices:
                if i in self.valid_indices:
                    return self.dataset[i]
                else:
                    try:
                        sample = self.dataset[i]
                        if self._is_valid_sample(sample):
                            self.valid_indices.add(i)
                            return sample
                        else:
                            self.invalid_indices.add(i)
                    except Exception:
                        self.invalid_indices.add(i)
        
        # 如果向前找不到，向后查找
        for i in range(current_idx - 1, -1, -1):
            if i not in self.invalid_indices:
                if i in self.valid_indices:
                    return self.dataset[i]
                else:
                    try:
                        sample = self.dataset[i]
                        if self._is_valid_sample(sample):
                            self.valid_indices.add(i)
                            return sample
                        else:
                            self.invalid_indices.add(i)
                    except Exception:
                        self.invalid_indices.add(i)
        
        # 如果都找不到，重新开始查找或返回第一个有效样本
        print(f"无法找到有效样本，当前索引: {current_idx}，重新开始查找")
        # 清空无效索引，重新开始
        self.invalid_indices.clear()
        # 尝试返回第一个样本
        try:
            sample = self.dataset[0]
            if self._is_valid_sample(sample):
                self.valid_indices.add(0)
                return sample
        except Exception:
            pass
        
        # 如果还是找不到，返回一个空样本或继续循环
        print("警告：无法找到任何有效样本，返回空样本")
        return {"jpg": None, "cls": 0}  # 返回一个默认的空样本

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

imagenet_transform_val = pth_transforms.Compose([
    pth_transforms.Resize(384, max_size=None),
    pth_transforms.CenterCrop(384),
    pth_transforms.ToTensor(),
    pth_transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # 将单通道转换为三通道
])

def safe_image_transform(img, is_train=True):
    """安全地处理图像，跳过有EXIF错误的图像"""
    try:
        # 首先安全地加载图像
        safe_img = safe_load_image(img)
        if safe_img is None:
            return None
        
        # 转换为RGB模式
        if safe_img.mode != "RGB":
            safe_img = safe_img.convert("RGB")
        
        # 根据是否为训练集选择不同的变换
        if is_train:
            return imagenet_transform_train(safe_img)
        else:
            return imagenet_transform_val(safe_img)
    except (UnicodeDecodeError, OSError, ValueError) as e:
        # 跳过有问题的图像
        print(f"跳过有问题的图像: {e}")
        return None

def collate_fn_imagenet_wds(batch, is_train=True):
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

            img = safe_image_transform(img, is_train=is_train)
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
    labels = torch.tensor(labels, dtype=torch.int32)
    
    return {"pixel_values": pixel_values, "labels": labels}

def collate_fn_imagenet_wds_train(batch):
    return collate_fn_imagenet_wds(batch, is_train=True)

def collate_fn_imagenet_wds_val(batch):
    return collate_fn_imagenet_wds(batch, is_train=False)

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
            collate_fn  = collate_fn_imagenet_wds_train,
            shuffle     = True,
            num_workers = config.num_workers,
            drop_last   = True,
            persistent_workers = True if config.num_workers > 0 else False,  # 保持worker进程以避免重复初始化
        )
        
        return dataloader
    elif config.name == "hybrid":
        data_files = []
        for path in config.train_path:
            data_files.extend(glob.glob(os.path.join(path, "*.tar")))
        print(f"Found {len(data_files)} tar files")

        dataset = load_dataset(
            "webdataset",
            data_files = data_files,
            split      = "train",
            num_proc   = 8,
            streaming  = False,
        )

        safe_dataset = SafeImageDataset(dataset)
        
        # 对于 IterableDataset，使用不同的配置
        if safe_dataset._is_iterable:
            dataloader = DataLoader(
                safe_dataset,
                batch_size  = config.batch_size,
                collate_fn  = collate_fn_imagenet_wds_train,
                shuffle     = False,  # IterableDataset 不支持 shuffle
                num_workers = 0,      # IterableDataset 通常不使用多进程
                drop_last   = False,
                persistent_workers = False,
            )
        else:
            dataloader = DataLoader(
                safe_dataset,
                batch_size  = config.batch_size,
                collate_fn  = collate_fn_imagenet_wds_train,
                shuffle     = True,
                num_workers = config.num_workers,
                drop_last   = True,
                persistent_workers = True if config.num_workers > 0 else False,
            )
        
        return dataloader

def get_dataloader_test(config):
    data_files = []
    for path in config.train_path:
        data_files.extend(glob.glob(os.path.join(path, "*.tar")))
    print(f"Found {len(data_files)} tar files")

    ds = load_dataset("webdataset", data_files=data_files, split="train", streaming=True)
    ds = SafeImageDataset(ds)
    dataloader = DataLoader(ds, batch_size=256, num_workers=8, collate_fn=collate_fn_test, drop_last=True, persistent_workers=True)

    return dataloader


def collate_fn_test(batch):
    pixel_values = []
    labels = []
    try:
        for sample in batch:
            jpg_data = sample["jpg"]
            pixel_value = imagenet_transform_train(jpg_data)
            pixel_values.append(pixel_value)
            labels.append(sample["cls"])
    except Exception as e:
        print("处理样本时出错", e)

    if len(pixel_values) == 0:
        print(f"批次中所有图像都被跳过")
        return {"pixel_values": torch.empty(0, 3, 384, 384), "labels": torch.empty(0)}
    
    pixel_values = torch.stack(pixel_values, dim=0)
    labels = torch.tensor(labels, dtype=torch.int32)
    
    return {"pixel_values": pixel_values, "labels": labels}

def get_imagenet_wds_val_dataloader(config):
    data_files = glob.glob(os.path.join("/data/phd/jinjiachun/dataset/timm/imagenet-1k-wds", "*validation*.tar"))

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
        batch_size  = 1,
        collate_fn  = collate_fn_imagenet_wds_val,
        shuffle     = False,
        num_workers = config.num_workers,
        drop_last   = False,
        persistent_workers = True if config.num_workers > 0 else False,  # 保持worker进程以避免重复初始化
    )

    return dataloader