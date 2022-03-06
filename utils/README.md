# Utils
* Naver Clova Face API
* Face Crop

## 1. [Naver Clover Face API]()
- `naver-api-1.ipynb` (22.02.22.화)
    - 총 2700장의 `normal` 이미지에 대해 다음의 정보를 저장한 데이터프레임을 생성한다.
    - gender 관련 정보
        - `gender_data` - 라벨링 된 성별 정보
        - `gender_api` - API로 예측한 성별 정보
        - `gender_conf_api` - API를 이용한 성별 예측의 confidence 값
    - age 관련 정보
        - `age_data` - 라벨링 된 연령 정보
        - `age_group_data` - 라벨링 된 연령대 정보
        - `age_api` - API로 예측한 연령 정보
        - `age_group_api` - API로 예측한 연령대 정보
        - `age_conf_api` - API를 이용한 연령 예측의 confidence 값
    - 기타 정보
        - `label` - 라벨링 된 최종 클래스 정보
        - `path` - 이미지 경로
- `*.pkl` - API로 예측한 결과를 저장한 데이터프레임
    - `XXXX-XXXX_naver-api.pkl` - API 적용을 부분적으로 나누어 진행한 결과
    - `no_mask_api_final.pkl` - 모든 데이터프레임을 하나로 저장한 결과
- `naver-api-2.ipynb` (22.02.23.수)
    - 라벨링 된 성별, 연령 정보와 API가 예측한 성별, 연령 정보가 다른 데이터에 대해 시각화한다.
    - 라벨링이 잘못된 경우나 판단이 어려울 것으로 보이는 데이터의 경향성을 파악한다.
***
## 2. [Face Crop]()
### OpenCV Haar Cascade를 이용한 얼굴 검출       
![image](https://user-images.githubusercontent.com/39791467/156918438-f765b965-696c-443a-935d-5a1da2691dfc.png)
1. 얼굴이 검출되면 눈이 있는지 확인하고 crop
2. 얼굴이 여러 개 검출되면 눈이 있는지 확인하고 가장 가능성 있는 부분 crop
3. 눈의 개수가 2개가 아니면 잘못 찾았다고 판단하고 정해진 부분 고정된 크기로 crop
4. 얼굴이 하나도 검출되지 않으면 정해진 부분 고정된 크기로 crop
***
