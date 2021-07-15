import os

import cv2
from torch.utils import data
from torchvision import transforms


class Dataset(data.Dataset):
    def __init__(self,
                 file_list,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        self.file_list = [line.strip().split() for line in open(file_list)]
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        item = self.file_list[index]
        name = os.path.splitext(os.path.basename(item[0]))[0]
        image = self.input_transform(cv2.imread(item[0]))
        label = self.input_transform(cv2.imread(item[1]))
        return image, label, name

    def input_transform(self, image):
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.mean,
                std=self.std,
            ),
            transforms.Resize((384, 512)),
        ])
        return image_transform(image)