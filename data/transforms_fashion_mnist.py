import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_transforms():
    return A.Compose([
        # A.Rotate(limit=5, p=0.05),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.05),
        # A.Normalize(mean=(0.1307,), std=(0.3081,)),
        ToTensorV2()  # Convert image to PyTorch tensor
    ])

def read_image(image):
    return np.array(image)  # Convert PIL image to numpy array
