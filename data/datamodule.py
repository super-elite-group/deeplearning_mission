import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms as T
from torchvision.datasets import MNIST
from data.transforms import get_transforms, read_image
import os
import cv2
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root # 이미지 파일이 저장된 디렉토리
        self.transform = transform # 이미지 전환 파이프라인
        self.images = os.listdir(root) # 디렉토리 내의 모든 이미지 파일을 리스트로 저장

    def __len__(self):
        return len(self.images) # 데이터셋의 총 데이터 수

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx]) # 이미지 파일 경로
        image = read_image(img_path)
        if self.transform: # 이미지 전환 파이프라인이 존재하면
            augmented = self.transform(image=image) # 이미지에 전환 파이프라인 적용
            image = augmented['image'] # 전환된 이미지
        image = T.ToTensor()(image)  # 이미지를 텐서로 변환
        return image

class MyDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super(MyDataModule, self).__init__()
        self.batch_size = config.batch_size # 배치 크기 현재 32
        self.download_root = config.download_root # MNIST 데이터셋 다운로드 경로 현재 './data'
        self.num_workers = config.num_workers # 데이터 로드에 사용할 프로세스 수 현재 15
        self.transform = get_transforms() # 이미지 전환 파이프라인


    def setup(self, stage=None):  # 데이터셋을 설정하는 메서드
        train_dataset = datasets.MNIST(self.download_root, train=True, download=True, transform=T.ToTensor())  # MNIST 훈련 데이터셋 다운로드 및 설정
        self.train = train_dataset  # 전체 훈련 데이터셋을 사용

        test_dataset = datasets.MNIST(self.download_root, train=False, download=True, transform=T.ToTensor())  # MNIST 테스트 데이터셋 다운로드 및 설정


        val_size = int(0.5 * len(test_dataset))  # 테스트 데이터셋의 50%를 검증 데이터셋으로 사용
        test_size = len(test_dataset) - val_size  # 나머지 50%를 테스트 데이터셋으로 사용
        self.val, self.test = random_split(test_dataset, [val_size, test_size])  # 검증 데이터셋과 테스트 데이터셋으로 나눔

        # Replace with custom dataset if using your own images
        # self.train = CustomDataset(root='./path/to/train', transform=self.transform)
        # self.val = CustomDataset(root='./path/to/val', transform=self.transform)
        # self.test = CustomDataset(root='./path/to/test', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers) # 훈련 데이터 로더

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers) # 검증 데이터 로더

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers) # 테스트 데이터 로더
