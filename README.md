# 프로젝트 개요
<img width="2146" height="154" alt="image" src="https://github.com/user-attachments/assets/c6ded399-d0f2-4333-add8-7d878aea28e5" />
<img width="1710" height="751" alt="image" src="https://github.com/user-attachments/assets/9c4c220d-b2cf-412e-8e96-23fae31dc326" />

# 데이터셋 수집 및 전처리 
- YOLOv8 Transfer-Learning 진행을 위한 데이터 수집 및 전처리
- 상품 식별이 어려운 이미지에 대해서는 라벨링 제외
<img width="1710" height="647" alt="image" src="https://github.com/user-attachments/assets/41a40d3b-04af-4313-acd8-a187bbba29cd" />

- 데이터 증강(Augmentation)을 통한 학습 데이터 확장
- 적용 이유 : 적은 데이터셋으로 학습 하였을 때 성능이 좋지 않음
- 139장 → 418장
<img width="1710" height="545" alt="image" src="https://github.com/user-attachments/assets/2cc2cb30-36d4-4aa4-8eb4-147c81b6c617" />

# 상품명 인식(Downstream task)
- 객체 인식 후 상품을 확인하기 위해 영상처리 사용
- 다양한 Feature Detector + Matcher 조합 실험 진행 
<img width="1479" height="497" alt="image" src="https://github.com/user-attachments/assets/1dd55d76-edcd-4cd8-812b-b69b5f6ce53a" />

- 실험 진행 결과 SURF+BFMatcher, SURF+FLANNMatcher의 조합이 가장 성능이 높음
- 아래의 표는 객체 인식 결과가 데이터베이스(DB)와 얼마나 일치하는지를 정확도(Accuracy) 기준으로 평가한 표
<img width="1844" height="458" alt="image" src="https://github.com/user-attachments/assets/77bb5dca-5dd8-455e-aec1-f68e73df0f58" />


# 결론 
- 데이터 수집 시 고려해야 할 다양한 요소들에 대해 사전에 계획을 세우는 것의 중요성 파악
- 데이터의 다양성과 일반화 성능을 확보하기 위한 데이터 증강 기법 학습
- 영상 처리 과정에서의 특징점 추출(Feature Detector) 및 매칭(Matcher) 알고리즘의 원리와 활용 방법에 대해 학습
<img width="910" height="494" alt="image" src="https://github.com/user-attachments/assets/251e6398-c731-4abc-a20f-052fbf18b72c" />

# Paper
(KCI) J. Si, D. Kim, S. Kim, “Automation of Online to Offline Stores: Extremely Small Depth-Yolov8 and Feature-Based Product Recognition”, The Journal of Korea Institute of Information, Electronics, and Communication Technology, Vol.17, No.3, pp.121-129, Jun. 2024.

