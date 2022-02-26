import timm
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

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
        self._criterion = loss.create_criterion(self.config.loss)
        self.f1_score = torchmetrics.F1Score(num_classes=self.config.model.num_class, average='macro', mdmc_average='global')

    def forward(self, x):
        return self.feature(x)

    def training_step(self, batch, batch_idx):
        preds, loss, acc, labels = self.__share_step(batch, 'train')
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        return {"loss": loss, "pred": preds.detach(), 'labels': labels.detach()}

    def validation_step(self, batch, batch_idx):
        preds, loss, acc, labels = self.__share_step(batch, 'val')
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        return {"pred": preds.detach(), 'labels': labels.detach()}

    def __share_step(self, batch, mode):
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
            self.parameters(), **self.config.optim.params)
        scheduler = eval(self.config.sche.name)(
            optimizer,
            **self.config.sche.params
        )
        return [optimizer], [scheduler]
