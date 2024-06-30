import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    # BaseTrainer 클래스를 상속받아 초기화
    # 파라미터들은 실제 사용하는 train.py 47줄에서 정의 
    # 예를 들어 optimizer = torch.optim, data_loader = config.init_obj('data_loader', module_data)
    # valid_data_loader = data_loader.split_validation(),lr_scheduler = torch.optim.lr_scheduler)
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None): # torch.optim.lr_scheduler
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch # len_epoch의 역할 : 한 에포크 내에서 총 몇 번의 배치를 처리할지를 나타냄. 보통 self.data_loader의 길이로 설정됨. 즉, 한 에포크 동안 데이터 로더가 제공할 수 있는 총 배치 수
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        #학습 및 검증 메트릭을 추적하는 MetricTracker(util.py) 객체를 생성. # reset, update, avg, result 함수 제공
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    # 한 에포크 동안 모델을 학습시키는 함수
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train() # 모델을 학습 모드로 설정
        self.train_metrics.reset() # 학습 메트릭 초기화

        """
        배치(batch): 한 번에 모델에 입력되는 데이터 묶음. 예를 들어, 배치 크기가 32이면, 한 배치에는 32개의 데이터 포인트가 포함됨
        에포크(epoch): 전체 데이터셋을 한 번 완전히 학습하는 과정
        """
        # '배치마다' 루프를 돈다. # 데이터 로더마다 루프를 도는데, data_loader는 dataLoader로 불러온 객체이고, 이는 배치마다 묶기 때문에.
        for batch_idx, (data, target) in enumerate(self.data_loader): 
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad() # 옵티마이저의 기울기 초기화
            output = self.model(data) # 모델의 결과 출력
            loss = self.criterion(output, target) # 손실 계산
            loss.backward() # 역전파를 통해 기울기를 계산한 후. nn의 함수
            self.optimizer.step() # 옵티마이저로 모델 파라미터 업데이트. # step할 때 계산한 것들이 업데이트 됨.

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx) # 현재 스텝 설정 # = 여태까지 총 진행된 배치 수
            self.train_metrics.update('loss', loss.item()) # 손실메트릭 업데이트
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            # 학습 상태를 로깅, 입력이미지를 텐서보드에 추가
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            """
            len_epoch의 역할 : 한 에포크 내에서 총 몇 번의 배치를 처리할지를 나타냄. 
            보통 self.data_loader의 길이로 설정됨. 
            즉, 한 에포크 동안 데이터 로더가 제공할 수 있는 총 배치 수
            """
            if batch_idx == self.len_epoch: # 모든 배치 다 돌면 루프 나오기
                break
        log = self.train_metrics.result() # 에포크가 끝나면 학습 메트릭을 로깅

        if self.do_validation: # 검증을 수행할지 여부를 나타내는 T/F 값
            val_log = self._valid_epoch(epoch) # _valid_epoch 메서드를 호출하여 검증을 수행하고, 검증 결과를 val_log에 저장
            log.update(**{'val_'+k : v for k, v in val_log.items()}) # 검증 결과를 로그에 업데이트

        if self.lr_scheduler is not None:
            self.lr_scheduler.step() # learning_rate 업데이트
        return log

    # 이 메서드는 한 에포크 동안 모델을 검증하는 함수
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval() # 모델을 평가 모드로 설정
        self.valid_metrics.reset() # 검증 메트릭 초기화
        with torch.no_grad(): # 검증 단계이기 때문에 불필요한 자동미분 로직 비활성화(메모리절약)
            for batch_idx, (data, target) in enumerate(self.valid_data_loader): # 매 배치마다 루프
                data, target = data.to(self.device), target.to(self.device) # GPU로 계산하겠다.

                output = self.model(data) # 결과 출력
                loss = self.criterion(output, target) # 손실 계산

                # 텐서보드에 손실, 메트릭값 기록 및 입력 이미지 시각화
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    # 진행상황 프로그래스바
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
