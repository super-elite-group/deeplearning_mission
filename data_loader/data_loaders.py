from torchvision import datasets, transforms
from base import BaseDataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import Dataset

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


# Arbumentation
# class MnistDataset(Dataset):
#     def __init__(self, mnist_dataset, transform=None):
#         self.mnist_dataset = mnist_dataset
#         self.transform = transform

#     def __len__(self):
#         return len(self.mnist_dataset)

#     def __getitem__(self, idx):
#         image, label = self.mnist_dataset[idx]
#         if self.transform:
#             image = np.array(image)  # PIL 이미지를 Numpy 배열로 변환
#             augmented = self.transform(image=image)
#             image = augmented['image']
#         return image, label
    
# class MnistDataLoader(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#         if training:
#             transform = A.Compose([
#                 A.RandomRotate90(),
#                 A.Transpose(),
#                 A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
#                 A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, p=0.5),
#                 A.Normalize(mean=(0.1307,), std=(0.3081,)),
#                 ToTensorV2()
#             ])
#         else:
#             transform = A.Compose([
#                 A.Normalize(mean=(0.1307,), std=(0.3081,)),
#                 ToTensorV2()
#             ])

#         self.data_dir = data_dir
#         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=None)
#         self.dataset = MnistDataset(self.dataset, transform=transform)

#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
