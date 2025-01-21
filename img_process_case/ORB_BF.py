from ultralytics import YOLO
import os
import cv2
import json
import numpy as np
import matplotlib.pylab as plt
from PIL import Image, ImageFont, ImageDraw
import json
import math
import cv2
import os

def compare_ftdetect(img, folder_path):    

    # 폴더 내의 모든 이미지 파일 경로 수집
    template_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            img_path = os.path.join(root, file)
            template_paths.append(img_path)
    
    #유사도 점수 저장할 딕셔너리 초기화
    similarity_scores = {}
    #gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for template_path in template_paths:
        #sift = cv2.xfeatures2d.SURF_create(hessianThreshold=1000, nOctaves=3, extended=True, upright=True)
        sift = cv2.ORB_create()
        img2 = imread(template_path)
        img2 = cv2.resize(img2, (640, 640))
        #gray2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # 주어진 이미지와 템플릿 이미지의 특징점 및 디스크립터를 계산
        kp1, des1 = sift.detectAndCompute(img, None) # 마스크를 사용하지 않아서 None
        kp2, des2 = sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher() #Brute-Force Mathcer를 생성하고 디스크립터 매칭
        # 인덱스 파라미터와 검색 파라미터 설정 ---①
        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # search_params = dict(checks=50)

        # Flann 매처 생성 ---③
        #matcher = cv2.FlannBasedMatcher(index_params, search_params)


        matches = matcher.knnMatch(des1, des2, 2)
        
        # 좋은 매칭 결과 선별
        good_matches = [] 
        for m in matches: # matches는 두개의 리스트로 구성
            if m[0].distance / m[1].distance <0.7: # 임계점 0.7
                good_matches.append(m[0]) 
        similarity_scores[template_path] = sum(match.distance for match in good_matches)
    most_similar_template = max(similarity_scores, key=similarity_scores.get)
    most_similar_folder = os.path.dirname(most_similar_template)
    predictName = os.path.basename(most_similar_folder)

    return predictName

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8): 
    try: 
        n = np.fromfile(filename, dtype) 
        img = cv2.imdecode(n, flags) 
        return img 
    except Exception as e: 
        print(e) 
        return None

snack_group=dict()

if __name__ == '__main__':
    model = YOLO('C:/Users/iialab/Desktop/o2o/v6/runs/detect/train/weights/best.pt')
    test_folder = 'C:/Users/iialab/Desktop/o2o/god/test/images/'
    name_folder = 'C:/Users/iialab/Desktop/o2o/total_ramen_name_rm/'  # 이름만 모아 놓은 곳
    #name_folder = 'C:/Users/iialab/Desktop/o2o/total_name_2/'  # 이름만 모아 놓은 곳
    #name_folder = 'C:/Users/iialab/Desktop/o2o/total_name_3/'  # 이름만 모아 놓은 곳

    image_files = os.listdir(test_folder)
    for image_file in image_files:
        image_path = os.path.join(test_folder, image_file)

        # 이미지에 대한 객체 감지 수행
        results = model.predict(image_path)

        # 결과 이미지 그리기
        img = imread(image_path)
        if img is None:
            print(f"오류: 이미지를 찾을 수 없거나 로드할 수 없습니다 - {image_path}")
            continue

        results_json = results[0].tojson()
        results_dict = json.loads(results_json)
        # 결과를 confidence 값을 기준으로 정렬
        results_dict.sort(key=lambda x: x['confidence'], reverse=True)

        snack_dict = {}
        name_dict = {}

        num = 0
        # 0 : name 1 : snack
        # 진열대는 반대
        for pred in results_dict:
            cls = int(pred['class'])
            conf = pred['confidence']
            if cls == 0:
                x1 = int(pred['box']['x1'])
                y1 = int(pred['box']['y1'])
                x2 = int(pred['box']['x2'])
                y2 = int(pred['box']['y2'])
                snack_dict[str(num)] = [(x1, y1), (x2, y2), conf]
                num += 1
        for snack in snack_dict.keys():
            
            left = snack_dict[snack][0][0]
            top = snack_dict[snack][0][1]
            right = snack_dict[snack][1][0]
            bottom = snack_dict[snack][1][1]
            modify_img = img[top:bottom, left:right].copy()
            
            predictName = compare_ftdetect(modify_img, name_folder)
            
            # x1, y1 정의
            x1 = left
            y1 = top
            
            fontpath = "C:/Users/iialab/Desktop/o2o/fonts/paper_gulim.ttc"
            font = ImageFont.truetype(fontpath, 20)
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((x1, y1 - 10), predictName, font=font, fill=(0, 255, 0, 2))
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            
        print(snack_dict)
       
        cv2.imshow('Object Detection', img)
        drawing = False
        start_x, start_y, end_x, end_y = -1, -1, -1, -1

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC 키를 누르면 종료
                break
        
        cv2.waitKey(0)

    cv2.destroyAllWindows() 