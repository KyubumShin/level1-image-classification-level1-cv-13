import os
import gc

import pandas as pd
import yaml
import argparse
from box import Box
from pprint import pprint

from sklearn.model_selection import StratifiedKFold
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


def __share_fit(config, transform, train_data: pd.DataFrame, val_data: pd.DataFrame = None, is_fold: bool = True):
    datamodule = CustomDataLoader(train_df=train_data, val_df=val_data, cfg=config, transform=transform)
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = None
    if is_fold:
        val_dataloader = datamodule.val_dataloader()
    model = ClassificationModel(config)
    callbacks = load_callback(is_fold)
    logger = TensorBoardLogger(config.log_name)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.epoch,
        callbacks=callbacks,
        auto_lr_find=True,
        **config.trainer,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    torch.cuda.empty_cache()
    gc.collect()


def train(config, arg):
    torch.cuda.empty_cache()
    gc.collect()
    is_fold = False if config.n_splits == 0 else True

    df = load_df(config)
    transform = CustomTransform()
    config.train_transform = str(transform.train_transform())
    if is_fold:
        skf = StratifiedKFold(
            n_splits=config.n_splits, shuffle=True, random_state=config.seed
        )
        config.val_trainform = str(transform.val_transform())
    config.to_yaml(os.path.join(os.getcwd(), cfg.log_name, "config.yaml"), sort_keys=False)

    if is_fold:
        for fold, (train_idx, val_idx) in enumerate(skf.split(df["id"], df["age"])):
            train_df = df.loc[train_idx].reset_index(drop=True)
            val_df = df.loc[val_idx].reset_index(drop=True)
            __share_fit(config, transform, train_df, val_df)
            if arg.debug == 1:
                break
    else:
        __share_fit(config, transform, df, is_fold=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.yaml',
                        help='config.yaml directory')
    parser.add_argument('--epoch', '-e', type=int, default=None,
                        help='number of epochs to train (default: config value)')
    parser.add_argument('--cv', type=int, default=1,
                        help='train with cross validation (default: 1')
    parser.add_argument('--debug', type=int, default=0,
                        help='run for debug (use only 1 fold)')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = Box(cfg)
    if args.epoch is not None:
        cfg.epoch = args.epoch
    cfg.log_name += f'_{cfg.n_splits}fold' if args.cv == 1 else f'_nofold'
    if args.cv == 0:
        cfg.n_splits = 0
    if not os.path.exists(os.path.join(os.getcwd(), cfg.log_name)):
        os.makedirs(os.path.join(os.getcwd(), cfg.log_name))
    pprint(cfg)
    seed_everything(cfg.seed)
    train(cfg, args)
