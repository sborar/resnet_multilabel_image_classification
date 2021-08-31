from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, csv, train=False, maj_frac=None, struc_list=None):
        if maj_frac:
            self.csv = self._sample(csv, maj_frac, struc_list)
        else:
            self.csv = csv
        self.all_image_names = self.csv[:]['img']
        self.all_labels = np.array(self.csv.drop(['img'], axis=1))
        self.train = train

        print(f"Number of images: {len(self.csv)}")
        self.image_names = list(self.all_image_names)
        self.labels = list(self.all_labels)
        # define the training transforms
        self.train_transform = A.Compose([
            A.Resize(128, 128, p=1),
            A.RandomScale(scale_limit=(0.75, 1.5), p=0.3),
            A.PadIfNeeded(min_height=128, min_width=128, p=1),
            A.CenterCrop(128, 128, p=1),
            A.RandomBrightnessContrast(p=0.2),
            A.Blur(p=0.2),
            A.Sharpen(p=0.2),
            A.Normalize(mean=(0.28, 0.28, 0.28),
                        std=(0.031, 0.031, 0.031),
                        max_pixel_value=1.0, p=1),
            ToTensorV2(always_apply=True)
            # normalize should be the same, get mean and standard deviation of the data we have
        ])

        self.test_transform = A.Compose([
            A.Resize(128, 128, p=1),
            A.Normalize(mean=(0.28, 0.28, 0.28),
                        std=(0.031, 0.031, 0.031),
                        max_pixel_value=1.0, p=1),
            ToTensorV2(always_apply=True)
        ])

    def _sample(self, df, frac, struc_list=None):
        if struc_list:
            struc = df[struc_list]
            df_bg = df[(struc == -1).any(axis=1)] # decrease how much im unde®smapling it
        else:
            df_bg = df[(df.iloc[:, 1:] == 1).any(axis=1)]
        df_struc = df[(df.iloc[:, 1:] == 1).any(axis=1)] # select based on some roi, change loss threshold, oversample some roi, try based on some roi
        print('imbalance', len(df_struc), len(df_bg))
        return df_bg.append(df_struc).reset_index(drop=True)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        # image = cv2.imread(self.image_names[index])
        x= Image.open(self.image_names[index])
        image = np.float32(np.array(x)/255.0)
        assert image is not None
        if self.train:
            image = self.train_transform(image=image)['image']
        else:
            image = self.test_transform(image=image)['image']
        targets = self.labels[index]

        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }
