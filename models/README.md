# Model
* PLModel

## 1. [pytorch_lightning_baseline Ver. ipynb](./pytorch_lightning_baseline_ipynb)
### Pytorch Lightning으로 제작한 Baseline 
Config의 mask로 모델 hyper parameter를 조절가능하다


## 2.  [pytorch_lightning_baseline Ver. python](./pytorch_lightning_baseline_IDE)
```
python train.py --config [config 경로] --cv [CV 사용여부 True/False] -e [Epoch 직접 수정할지]
```
* code폴더에 파일을 넣을경우 config에서 image_dir과 data_dir의 경로를 아래와 같이 수정해주어야 합니다.
**../../input -> ../input**
* 세부 파라미터는 config.yaml을 통한 수정해주세요  
* inference 파일은 제작중입니다  
* 저장폴더명에 자동으로 fold를 넣게 바꾸었기 때문에 폴더관리가 조금더 쉬워졌습니다