import os
from albumentations.augmentations.geometric.resize import Resize
from albumentations.augmentations.transforms import Normalize

import cv2
from torch.utils import data
from torchvision import transforms
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2


# 512_images mean: [0.18666788 0.20928789 0.19854261]
#            std: [0.2281134  0.23495037 0.23019542]

# 204 images mean: [0.17638991 0.19891686 0.18880841]
#            std: [0.2221894  0.22992185 0.22523154]
class Dataset(data.Dataset):
    def __init__(self,
                 file_list,
                 mean=[0.176, 0.199, 0.189],
                 std=[0.222, 0.223, 0.225]):
        self.file_name = file_list
        self.file_list = [line.strip().split() for line in open(file_list)]
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        item = self.file_list[index]
        name = os.path.splitext(os.path.basename(item[0]))[0]
        image = cv2.imread(item[0])
        if "test" in self.file_name:
            augment = self.testset_transform(image)
            return augment["image"]/255, name
        else:
            label = cv2.imread(item[1])
            augment = self.transform(image, label)
            return augment["image"]/255, augment["label"]/255, name

    def transform(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        input_transform = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.Rotate(limit=90, p=0.5),
            # albu.PadIfNeeded(min_height=192, min_width=192, border_mode=0, p=0.5),
            # albu.RandomCrop(192, 192),
            # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, border_mode=0, p=0.5),

            # albu.GaussNoise(p=0.2),
            # albu.Perspective(p=0.5),

            # albu.OneOf(
            #     [
            #         albu.CLAHE(p=1),
            #         albu.RandomGamma(p=1),
            #     ],
            #     p=0.5,
            # ),

            # albu.OneOf(
            #     [
            #         albu.Sharpen(p=1),
            #         albu.Blur(blur_limit=3, p=1),
            #         albu.MotionBlur(blur_limit=3, p=1),
            #     ],
            #     p=0.5,
            # ),

            # albu.OneOf(
            #     [
            #         albu.RandomBrightnessContrast(p=1),
            #         albu.HueSaturationValue(p=1),
            #     ],
            #     p=0.5,
            # ),
            # albu.Normalize(
            #     mean=self.mean,
            #     std=self.std
            # ),
            ToTensorV2(transpose_mask=True),
        ],
        additional_targets={"label": "image"})
        return input_transform(image=image, label=label)

    def testset_transform(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        test_transform = albu.Compose([
            ToTensorV2(transpose_mask=True),
        ])
        return test_transform(image=image)