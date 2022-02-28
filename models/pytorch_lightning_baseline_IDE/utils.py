import pandas as pd
import torchvision.transforms as transforms
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar

def load_df(config) -> pd.DataFrame:
    df = pd.read_csv(config.data_dir)
    df = fix_data(df)
    return df


def fix_data(df: pd.DataFrame) -> pd.DataFrame:
    errors = ["006359", "006360", "006361", "006362", "006363", "006364"]
    for error in errors:
        df.loc[df["id"] == error, "gender"] = "male"
    return df


def load_callback(is_fold: bool = True) -> list:
    early_stopping = EarlyStopping(monitor="val_f1_score", mode='max', patience=3)
    lr_monitor = callbacks.LearningRateMonitor()
    score_checkpoint = callbacks.ModelCheckpoint(
        filename="best_score",
        monitor="val_f1_score",
        save_top_k=2,
        mode="max",
        save_last=False,
    )
    loss_checkpoint = callbacks.ModelCheckpoint(
        filename="best_loss",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=False,
    )
    tqdm_progressbar = TQDMProgressBar(
        refresh_rate=1,
    )
    if is_fold:
        return [early_stopping, lr_monitor, score_checkpoint, loss_checkpoint]
    return [lr_monitor, tqdm_progressbar]


def get_grad_cam(model):
    pass
