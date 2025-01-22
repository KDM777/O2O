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

    for template_path in template_paths:
        sift = cv2.xfeatures2d.SURF_create()

        img2 = imread(template_path)
        img2 = cv2.resize(img2, (640, 640))
        # 주어진 이미지와 템플릿 이미지의 특징점 및 디스크립터를 계산
        kp1, des1 = sift.detectAndCompute(img, None) # 마스크를 사용하지 않아서 None
        kp2, des2 = sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher() #Brute-Force Mathcer를 생성하고 디스크립터 매칭
        
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
    img_folder = 'C:/Users/iialab/Desktop/o2o/god/test/images/'
    label_folder='C:/Users/iialab/Desktop/o2o/god/test/labels/'
    name_folder = 'C:/Users/iialab/Desktop/o2o/paper_ftmatch/name_data/'  # 이름만 모아 놓은 곳
    #name_folder = 'C:/Users/iialab/Desktop/o2o/total_name_2/'  # 이름만 모아 놓은 곳
    #name_folder = 'C:/Users/iialab/Desktop/o2o/total_name_3/'  # 이름만 모아 놓은 곳

    image_files = os.listdir(img_folder)
    label_files = os.listdir(label_folder)

    for image_file, label_file in zip(image_files, label_files):
        image_path = os.path.join(img_folder, image_file)
        label_path = os.path.join(label_folder, label_file)

        # 결과 이미지 그리기
        img = imread(image_path)
        height, width, _ = img.shape

        if img is None:
            print(f"오류: 이미지를 찾을 수 없거나 로드할 수 없습니다 - {image_path}")
            continue
        data_list=[]

        with open(label_path, 'r') as file:
            for line in file:
                elements =line.strip().split()
                elements = list(map(float, elements))         
                cord=[]
                cord.append((elements[1]-(elements[3]/2))*width)
                cord.append((elements[2]-(elements[4]/2))*height)
                cord.append((elements[1]+(elements[3]/2))*width)
                cord.append((elements[2]+(elements[4]/2))*height)
                cord = list(map(int, cord))
                data_list.append(cord)

        snack_dict = {}
        name_dict = {}

        num = 0
        # 0 : name 1 : snack
        # 진열대는 반대
        for pred in data_list:
            x1 = int(pred[0])
            y1 = int(pred[1])
            x2 = int(pred[2])
            y2 = int(pred[3])
            snack_dict[str(num)] = [(x1, y1), (x2, y2)]
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
            
            fontpath = "C:/Users/iialab/Desktop/o2o/o2o_begin/fonts/gulim.ttc"
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