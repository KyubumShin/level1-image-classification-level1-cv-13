# Model
* PLModel

# 1.  [pytorch_lightning_baseline IDE](./pytorch_lightning_baseline_IDE)
## 1. Train
```
python train.py --config [config 경로] --cv [CV 사용여부 True/False] -e [Epoch 직접 수정할지]
```
* code폴더에 파일을 넣을경우 config에서 image_dir과 data_dir의 경로를 아래와 같이 수정해주어야 합니다.
**../../input -> ../input**
* 세부 파라미터는 config.yaml을 통한 수정해주세요  
* inference 파일은 제작중입니다  
* 저장폴더명에 자동으로 fold를 넣게 바꾸었기 때문에 폴더관리가 조금더 쉬워졌습니다
## 2. lr_find
```
python lr_fit.py --config [config 경로]
```
* pytorch lightning의 AutoML 기능을 이용한 최적 Learning Rate 탐색

## 3. Inference
```
python ensemble.py --config [ensemble config 경로]
```
* ensemble config 파일에 작성된 모든 모델로 soft voting ensemble을 진행

# 2. [pytorch_lightning_baseline ipynb](./pytorch_lightning_baseline_ipynb)
Pytorch Lightning으로 제작한 Baseline   
