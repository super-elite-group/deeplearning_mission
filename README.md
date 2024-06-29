# deeplearning_mission

## MNIST 데이터 셋

1. Pythorch Template 활용하여 MLP 모델 만들기 - [pythorch Template](https://github.com/victoresque/pytorch-template)
2. Pytorch 공식 튜토리얼 문서의 컴퓨터 비전 전이 학습을 이해하고 각 줄에 대한 주석 달기 - [pytorch 공식 튜토리얼](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
3. CNN 모델을 사용하여 99% 이상 성능 내는 모델 만들기
4. RNN 모델을 사용하여 98% 이상 성능 내는 모델 만들기
5. LSTM 모델을 사용하여 98% 이상 성능 내는 모델 만들기
6. GRU 모델을 사용하여 98% 이상 성능 내는 모델 만들기
7. Albumentation 라이브러리를 사용하여 데이터 증강하여 99.5% 이상의 성능을 내는 모델을 만들기 - [Albumentation 라이브러리](https://albumentations.ai/)
8. Convolution 과 Activation 레이어만 사용하여 MNIST 분류기 만들기 (FCLayer, Flatten 없이 구현)
9. Semi-supervised learning을 이용해서 MNIST 분류기 만들기 - [참고자료1](https://blog.est.ai/2020/11/ssl/), [참고자료2](https://github.com/rubicco/mnist-semi-supervised)

## MNIST 데이터 셋 만드는 코드

```
import torchvision.transforms as T
import torchvision
import torch
from torch.utils.data import DataLoader

download_root = './MNIST_DATASET'

mnist_transform = T.Compose([
    T.ToTensor(),
])

train_dataset = torchvision.datasets.MNIST(download_root, transform=mnist_transform, train=True, download=True)
test_dataset = torchvision.datasets.MNIST(download_root, transform=mnist_transform, train=False, download=True)

total_size = len(train_dataset)
train_num, valid_num = int(total_size * 0.8), int(total_size * 0.2)
train_dataset,valid_dataset = torch.utils.data.random_split(train_dataset, [train_num, valid_num])

batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
```
