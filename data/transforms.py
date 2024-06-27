import albumentations as A
import cv2

# Declare an augmentation pipeline
def get_transforms():
    return A.Compose([
        # A.RandomCrop(width=256, height=256), # 무작위로 이미지를 256x256 크기로 자름
        A.HorizontalFlip(p=0.5), # 50% 확률로 이미지를 좌우 반전
        A.RandomBrightnessContrast(p=0.2), # 20% 확률로 이미지의 밝기와 대비를 조정
    ])

def read_image(image_path):
    # image = cv2.imread(image_path) # 이미지 파일을 읽음
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 이미지 색상을 BGR에서 RGB로 변환
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # 이미지을 grayscale로 읽음
    return image
