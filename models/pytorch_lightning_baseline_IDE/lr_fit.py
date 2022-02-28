import os
import gc

import yaml
import argparse
from box import Box
from pprint import pprint

import numpy as np
import random
import torch

import pytorch_lightning as pl
from model import ClassificationModel
from dataset import CustomDataLoader, CustomTransform
from utils import load_df, load_callback
from pytorch_lightning.loggers import TensorBoardLogger


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main(config):
    torch.cuda.empty_cache()
    gc.collect()
    seed_everything(config.seed)
    df = load_df(config)
    transform = CustomTransform()
    config.train_transform = str(transform.train_transform())
    train_dataloader = CustomDataLoader(train_df=df, cfg=config, transform=transform).train_dataloader()
    model = ClassificationModel(config)
    callbacks = load_callback()
    logger = TensorBoardLogger(config.log_name)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.epoch,
        callbacks=callbacks,
        auto_lr_find=True,
        **config.trainer,
    )
    lr_finder = trainer.tuner.lr_find(model, train_dataloaders=train_dataloader)
    new_lr = lr_finder.suggestion()
    config.lr = new_lr
    config.to_yaml(os.path.join(os.getcwd(), "lr.yaml"), sort_keys=False)

    del trainer
    del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.yaml',
                        help='config.yaml directory')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = Box(cfg)
    pprint(cfg)
    seed_everything(cfg.seed)
    main(cfg)
