import timm
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

import torchmetrics
from torchmetrics.functional import accuracy
import pytorch_lightning as pl

import loss


class ClassificationModel(pl.LightningModule):
    def __init__(self, cfg):
        super(ClassificationModel, self).__init__()
        self.config = cfg
        self.feature = timm.create_model(self.config.model.name, pretrained=True,
                                         num_classes=self.config.model.num_class)
        self._criterion = loss.create_criterion(self.config.loss.name, **self.config.loss.params)
        self.f1_score = torchmetrics.F1Score(num_classes=self.config.model.num_class, average='macro', mdmc_average='global')
        self.learning_rate = self.config.lr


    def forward(self, x):
        return self.feature(x)

    def training_step(self, batch, batch_idx):
        preds, loss, acc, labels = self.__share_step(batch, 'train')
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return {"loss": loss, "pred": preds.detach(), 'labels': labels.detach()}
        #detach() : 기존 Tensor에서 gradient 전파가 안되는 텐서 생성
        #단 storage를 공유하기에 detach로 생성한 Tensor가 변경되면 원본 Tensor도 똑같이 변합니다.
        #clone() : 기존 Tensor와 내용을 복사한 텐서 생성

    def validation_step(self, batch, batch_idx):
        preds, loss, acc, labels = self.__share_step(batch, 'val')
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return {"loss": loss, "pred": preds, 'labels': labels}

    def rand_bbox(self, size, lam):  # size : [Batch_size, Channel, Width, Height]
            W = size[2]
            H = size[3]
            cut_rat = np.sqrt(1. - lam)  # 패치 크기 비율
            #cut_w = np.int(W * cut_rat) # vertical cutmix 라서 필요 없음
            cut_h = np.int(H * cut_rat)

            # 패치의 중앙 좌표 값 cx, cy
            #cx = np.random.randint(W) # vertical cutmix 라서 필요 없음
            cy = np.random.randint(H) #전체 위치에서 random 위치에 cut mix. 
            #cy = H//2 + cut_h//2 #얼굴 기준 중앙보다 조금 옆으로 이동한 후, cutmix 

            # 패치 모서리 좌표 값
            bbx1 = 0
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = W
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            return bbx1, bby1, bbx2, bby2


    def cutmix(self, data, targets ,alpha):
        lam = np.random.beta(alpha+2, alpha)
        #들어온 배치 전부다. 
        rand_index = torch.randperm(data.size(0)) #난수 순열

        #loss 계산을 위함 
        target_a = targets  # 원본 이미지 label
        target_b = targets[rand_index]  # 패치 이미지 label

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(data.size(), lam)
        #데이터에 난수 순열 순서로 잘라서 합치기 
        data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
        targets = (target_a, target_b, lam)

        return data, targets
 

    def __share_step(self, batch, mode):
        if mode == 'train' and np.random.random()>0.5: # cutmix 작동될 확률      
            x, y = batch
            x, targets = self.cutmix(x,y,1)
            y_a, y_b, lam = targets
            y_hat = self.feature(x)
            loss = lam * self._criterion(y_hat, y_a) + (1 - lam) * self._criterion(y_hat, y_b)
            y = torch.round(y_a *lam + y_b * (1-lam))
            y = y.to(torch.int)
        else:
            x, y = batch
            y_hat = self.feature(x)
            loss = self._criterion(y_hat, y)

        acc = accuracy(y_hat, y)
        return y_hat, loss, acc, y


    def training_epoch_end(self, outputs):
        self.__cal_metrics(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.__cal_metrics(outputs, 'val')

    def __cal_metrics(self, outputs, mode) -> None:
        """
        metrics 계산을 위한 함수
        params:
            outputs: 각 epoch에 나온 log(loss:only train_epoch, pred, labels)
            mode: train or validation?
        """
        preds = []
        labels = []
        losses = []
        for out in outputs:
            pred, label = out['pred'], out['labels']
            preds.append(pred)
            labels.append(label)
        cat_preds = torch.cat(preds)
        arg_max_preds = torch.argmax(cat_preds, dim=1)
        labels = torch.cat(labels)
        metrics = self.f1_score(arg_max_preds, labels)
        self.log(f"{mode}_f1_score", metrics)


    def configure_optimizers(self):
        optimizer = eval(self.config.optim.name)(
            self.parameters(), lr=self.learning_rate)
        scheduler = eval(self.config.sche.name)(
            optimizer,
            **self.config.sche.params
        )
        return [optimizer], [scheduler]

    
   