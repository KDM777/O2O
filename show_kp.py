import cv2
import os
import numpy as np

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8): 
    try: 
        n = np.fromfile(filename, dtype) 
        img = cv2.imdecode(n, flags) 
        return img 
    except Exception as e: 
        print(e) 
        return None
if __name__ == '__main__':

    feature_detector = {
        'SIFT': cv2.SIFT_create(nfeatures=100),
    }

    # 폴더 내의 모든 이미지 파일 경로 수집
    template_paths = []
    #for root, _, files in os.walk('C:/Users/iialab/Desktop/o2o/shelf_v1/im2/'):
    for root, _, files in os.walk('C:/Users/iialab/Desktop/o2o/v1/name/'):
        for file in files:
            img_path = os.path.join(root, file)
            template_paths.append(img_path)
    
    #유사도 점수 저장할 딕셔너리 초기화
    similarity_scores = {}

    for detector_type, detector in feature_detector.items():
        for template_path in template_paths:
            img2 = imread(template_path)
            img2 = cv2.resize(img2, (640, 640))
            # 주어진 이미지의 keypoint 탐지
            kp2 = detector.detect(img2)
            img_draw = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            
            cv2.imshow('sift', img_draw)
            cv2.waitKey(0)  # 키보드 입력을 기다림
    cv2.destroyAllWindows()  # 창 닫기