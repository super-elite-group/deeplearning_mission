import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import numpy as np

# Declare an augmentation pipeline
def get_transforms():
    # return A.Compose([
    #     # A.RandomCrop(width=256, height=256), # 무작위로 이미지를 256x256 크기로 자름
    #     A.HorizontalFlip(p=0.5), # 50% 확률로 이미지를 좌우 반전
    #     # A.RandomBrightnessContrast(p=0.2), # 20% 확률로 이미지의 밝기와 대비를 조정
    #     A.Normalize(mean=(0.5,), std=(0.5,)),  # Normalize images to mean 0.5 and std 0.5
    #     ToTensorV2()
    # ])

    return A.Compose([
        A.Rotate(limit=5, p=0.05),  # Rotate images within a range of ±10 degrees
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.05),  # Small translations and rotations
        A.Normalize(mean=(0.1307,), std=(0.3081,)),
        ToTensorV2()  # Convert image to PyTorch tensor
    ])


def read_image(image):
    # image = cv2.imread(image_path) # 이미지 파일을 읽음
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 이미지 색상을 BGR에서 RGB로 변환

    return np.array(image)  # Convert PIL image to numpy array
    # return image
