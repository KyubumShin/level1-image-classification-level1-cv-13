import os
import pandas as pd
import numpy as np
import argparse
import yaml
from box import Box
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from model import ClassificationModel
from dataset import CustomTransform, TestDataset


def __model_load(config_dir, model_dir, loader, target) -> np.ndarray:
    with open(os.path.join(config_dir), "r") as f:
        config = yaml.safe_load(f)
    config = Box(config)
    device = torch.device('cuda')
    model = ClassificationModel(config)
    model.load_state_dict(torch.load(model_dir)['state_dict'])
    model = model.to(device).eval()
    predictions = []
    for images in tqdm(loader, total=len(loader)):
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = torch.softmax(pred, dim=1)
            predictions.extend(pred.cpu().numpy())
    del model
    return np.array(predictions)


def main(cfg):
    test_dir = '/opt/ml/input/data/eval'
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')
    test_transform = CustomTransform().val_transform()
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    dataset = TestDataset(image_paths, test_transform)
    loader = DataLoader(
        dataset,
        shuffle=False,
    )
    for key, items in cfg.items():
        oof_pred = None
        split = len(items.model_dir)
        models_dir = items.model_dir
        configs_dir = items.config_dir
        for model_dir, config_dir in zip(models_dir, configs_dir):
            preds = __model_load(config_dir, model_dir, loader, key)
            if oof_pred is None:
                oof_pred = preds / split
            else:
                oof_pred += preds / split
        submission[key] = np.argmax(oof_pred, axis=1)
    submission.to_csv(os.path.join(test_dir, 'submission_test.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-C', type=str, default=None,
                        help='config directory')
    args = parser.parse_args()

    with open(os.path.join(args.config), "r") as f:
        cfg = yaml.safe_load(f)
    cfg = Box(cfg)
    main(cfg)
