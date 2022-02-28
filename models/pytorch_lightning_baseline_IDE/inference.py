import os
import pandas as pd
import argparse
import torch
import yaml
from box import Box
from tqdm import tqdm
import gc

from model import ClassificationModel
from torch.utils.data import DataLoader
from dataset import CustomTransform, TestDataset


def __model_load(config_dir, model_dir, df: pd.DataFrame, loader) -> pd.DataFrame:
    with open(os.path.join(config_dir), "r") as f:
        config = yaml.safe_load(f)
    config = Box(config)
    device = torch.device('cuda')
    model = ClassificationModel(config)
    model.load_state_dict(torch.load(model_dir)['state_dict'])
    model = model.to(device).eval()
    print("Done")
    predictions = []
    for images in tqdm(loader, total=len(loader)):
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            predictions.extend(pred.cpu().numpy())
    df[config.target] = predictions
    del model
    return df


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
        submission = __model_load(items.config_dir, items.model_dir, submission, loader)

    submission.to_csv(os.path.join(test_dir, 'submission_sample.csv'), index=False)
    submission['ans'] = submission['mask'] * 6 + submission['gender'] * 3 + submission['age_group']
    sub = submission.drop(columns=['age_group', 'mask', 'gender'])
    sub.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-C', type=str, default=None,
                        help='config directory')
    args = parser.parse_args()

    with open(os.path.join(args.config), "r") as f:
        cfg = yaml.safe_load(f)
    cfg = Box(cfg)
    main(cfg)
