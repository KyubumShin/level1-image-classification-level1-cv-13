# See:Real
## CV 13조
13조 See: Real의 Mask Image Classifiction


## Requirement
```
timm==0.5.4
torch==1.7.1
pytorch-lightning==1.5.10
python-box==5.4.1
torchmetrics==0.7.2
torchvision==0.8.2
adamp==0.3.0
```

## Train (IDE)
```
python train.py --config [CONFIG_DIR]
```
config는 첨부된 예시를 참고해주십시오

## Inference (IDE)
```
python ensemble.py --config [ENSEMBLE_DIR]
```
config는 첨부된 예시를 참고해주십시오
