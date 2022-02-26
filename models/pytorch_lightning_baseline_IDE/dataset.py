import os
from enum import Enum
from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class CustomDataset(Dataset):
    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    def __init__(self, df: pd.DataFrame, cfg, transform: transforms = None, mode: bool = True):
        self.config = cfg
        self.transform = transform
        self.data = self.__make_dataframe(df)
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __make_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        temp = pd.DataFrame(columns=['id', 'gender', 'age', 'age_group', 'mask', 'path', 'label'])
        for line in df.iloc:
            for file in list(os.listdir(os.path.join(self.config.data.image_dir, line['path']))):
                _file_name, ext = os.path.splitext(file)
                if _file_name not in self._file_names:
                    continue
                mask_label = self._file_names[_file_name]
                gender_label = GenderLabels.from_str(line['gender'])
                age_group = self.age_group(line['age'])
                data = {
                    'id': line['path'],
                    'gender': gender_label,
                    'age': line['age'],
                    'age_group': age_group,
                    'mask': mask_label,
                    'path': os.path.join(self.config.data.image_dir, line['path'], file),
                    'label': mask_label * 6 + gender_label * 3 + age_group
                }
                temp = temp.append(data, ignore_index=True)
        return temp

    def age_group(self, x):
        if x < self.config.data.min:
            return 0
        elif x < self.config.data.max:
            return 1
        else:
            return 2

    def __getitem__(self, index):
        data = self.data.iloc[index]
        img = read_image(data["path"])
        if self.transform:
            img = self.transform(img)

        y = None
        if self.mode:
            y = data[self.config.target]
        return img, y


class CustomDataLoader(LightningDataModule):
    def __init__(self, train_df: pd.DataFrame, transform: transforms = None,
                 val_df: pd.DataFrame = None, test_df: pd.DataFrame = None, cfg=None):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.config = cfg
        self.transform = transform

    def train_dataloader(self):
        train_dataset = CustomDataset(self.train_df, self.config, transform=self.transform.train_transform())
        return DataLoader(train_dataset, **self.config.dataloader)

    def val_dataloader(self):
        val_dataset = CustomDataset(self.val_df, self.config, transform=self.transform.val_transform())
        return DataLoader(val_dataset, **self.config.dataloader)


class CustomTransform:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.share_transform = transforms.Compose([
            transforms.CenterCrop([320, 320])
        ])
        self.mean = mean
        self.std = std

    def train_transform(self):
        return transforms.Compose([
            self.share_transform,
            transforms.RandomHorizontalFlip(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def val_transform(self):
        return transforms.Compose([
            self.share_transform,
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])