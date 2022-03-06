# Utils
* Naver Clova Face API
* Face Crop

## 1. [Naver Clover Face API]()
util에 대한 설명
***
## 2. [Face Crop]()
OpenCV Haar Cascade를 이용한 얼굴 검출
![image](https://user-images.githubusercontent.com/39791467/156918438-f765b965-696c-443a-935d-5a1da2691dfc.png)
1. 얼굴이 검출되면 눈이 있는지 확인하고 crop
2. 얼굴이 여러 개 검출되면 눈이 있는지 확인하고 가장 가능성 있는 부분 crop
3. 눈의 개수가 2개가 아니면 잘못 찾았다고 판단하고 정해진 부분 고정된 크기로 crop
4. 얼굴이 하나도 검출되지 않으면 정해진 부분 고정된 크기로 crop
***
