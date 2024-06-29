import pytorch_lightning as pl
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor
import numpy as np
import random
from data.transforms import get_transforms

class CustomFashionMNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, randomize=False):
        self.dataset = datasets.FashionMNIST(root, train=train, download=download)
        self.transform = transform
        self.target_transform = target_transform
        self.randomize = randomize

        if self.randomize:
            self.random_labels()

    def random_labels(self):
        labels = list(range(10))  # Assuming FashionMNIST has 10 classes
        random.shuffle(labels)
        self.randomized_label_map = {i: labels[i] for i in range(10)}
        print(labels)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        if self.randomize:
            label = self.randomized_label_map[label]

        if self.transform:
            image = np.array(image)[..., np.newaxis]  # Ensure the image has shape (28, 28, 1) 1 차원 추가
            transformed = self.transform(image=image)
            image = transformed["image"]

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, download_root, num_workers, randomize=True):
        super().__init__()
        self.batch_size = batch_size
        self.download_root = download_root
        self.num_workers = num_workers
        self.randomize = randomize

    def setup(self, stage=None):
        transform = get_transforms()

        self.train = CustomFashionMNISTDataset(
            root=self.download_root, train=True, transform=transform, download=True, randomize=self.randomize
        )
        test_transform = get_transforms()
        self.test = CustomFashionMNISTDataset(
            root=self.download_root, train=False, transform=test_transform, download=True, randomize=self.randomize
        )

        val_size = int(0.5 * len(self.test))
        test_size = len(self.test) - val_size
        self.val, self.test = random_split(self.test, [val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
