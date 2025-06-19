import os
import torch
import glob
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as pth_transforms

os.environ["TOKENIZERS_PARALLELISM"] = "false"


imagenet_transform_train = pth_transforms.Compose([
    pth_transforms.Resize(384, max_size=None),
    pth_transforms.RandomHorizontalFlip(p=0.5),
    pth_transforms.CenterCrop(384),
    pth_transforms.ToTensor(),
    pth_transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # 将单通道转换为三通道
])

def collate_fn_imagenet_wds(batch):
    pixel_values = []
    labels = []
    for sample in batch:
        img = imagenet_transform_train(sample["jpg"])
        if img.shape[0] != 3:
            print("skip", img.shape)
            continue
        labels.append(sample["cls"])
        pixel_values.append(img)
    
    pixel_values = torch.stack(pixel_values, dim=0)
    labels = torch.Tensor(labels)
    # print(pixel_values.shape, labels.shape)
    
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
