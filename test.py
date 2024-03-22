from ultralytics import YOLO
import os
import cv2
import json
import numpy as np
import matplotlib.pylab as plt
from PIL import Image, ImageFont, ImageDraw
import math

def compare_ftdetect(img, folder_path):    

    # 폴더 내의 모든 이미지 파일 경로 수집
    template_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            img_path = os.path.join(root, file)
            template_paths.append(img_path)
    
    # 유사도 점수 저장할 딕셔너리 초기화
    similarity_scores = {}

    for template_path in template_paths:
        sift = cv2.xfeatures2d.SURF_create(2000,8,8,True,False)

        img2 = imread(template_path)
        img2 = cv2.resize(img2, (640, 640))
        # 주어진 이미지와 템플릿 이미지의 특징점 및 디스크립터를 계산
        kp1, des1 = sift.detectAndCompute(img, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher() # Brute-Force Mathcer를 생성하고 디스크립터 매칭
        
        matches = matcher.knnMatch(des1, des2, 2)
        
        # 좋은 매칭 결과 선별
        good_matches = [] 
        for m in matches:
            if m[0].distance / m[1].distance < 0.7:
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

snack_group = {}

if __name__ == '__main__':
    img_folder = 'C:/Users/iialab/Desktop/o2o/god/demo/images/'
    label_folder = 'C:/Users/iialab/Desktop/o2o/god/demo/labels/'
    ramen_name_folder = 'C:/Users/iialab/Desktop/o2o/paper_ftmatch/name_data/ramen/'  # 이름만 모아 놓은 곳
    snack_name_folder = 'C:/Users/iialab/Desktop/o2o/paper_ftmatch/name_data/snack/'  # 이름만 모아 놓은 곳

    classes_dict = {}
    with open("C:/Users/iialab/Desktop/o2o/god/demo/labels/classes.txt", "r") as f:
        for i, line in enumerate(f):
            key = line.strip()
            classes_dict[key] = i
    
    image_files = os.listdir(img_folder)
    label_files = os.listdir(label_folder)
    total_num = 0
    total_precision = 0
    j = 0

    for image_file, label_file in zip(image_files, label_files):
        print(j)
        image_path = f"{img_folder}{j+1}.jpg"
        print(image_path)
        label_path = f"{label_folder}{j+1}.txt"

        img = imread(image_path)
        height, width, _ = img.shape

        if img is None:
            print(f"오류: 이미지를 찾을 수 없거나 로드할 수 없습니다 - {image_path}")
            continue
        data_list = []

        with open(label_path, 'r') as file:
            for line in file:
                elements = line.strip().split()
                elements = list(map(float, elements))
                cord = []
                cord.append((elements[1] - (elements[3] / 2)) * width)
                cord.append((elements[2] - (elements[4] / 2)) * height)
                cord.append((elements[1] + (elements[3] / 2)) * width)
                cord.append((elements[2] + (elements[4] / 2)) * height)
                cord.append(elements[0])
                cord = list(map(int, cord))
                data_list.append(cord)

        snack_dict = {}
        name_dict = {}
        num = 0
        precision = 0

        for pred in data_list:
            x1 = int(pred[0])
            y1 = int(pred[1])
            x2 = int(pred[2])
            y2 = int(pred[3])
            snack_dict[str(num)] = [(x1, y1), (x2, y2), pred[4]]
            num += 1
            total_num += 1
        for snack in snack_dict.keys():
            left = snack_dict[snack][0][0]
            top = snack_dict[snack][0][1]
            right = snack_dict[snack][1][0]
            bottom = snack_dict[snack][1][1]

            modify_img = img[top:bottom, left:right].copy()
            if j > 10:
                predictName = compare_ftdetect(modify_img, snack_name_folder)
            else:
                predictName = compare_ftdetect(modify_img, ramen_name_folder)
            predictNum = classes_dict.get(predictName, -1)
            print(predictNum, predictName, snack_dict[snack][2])
            if predictNum == snack_dict[snack][2]:
                precision += 1
                total_precision+=1
            
        print("맞는 개수 : %s, 개수 : %s, 정확도 : %s" % (precision, num, precision / num))
        j += 1

    print("전체 맞는 개수 : %s, 전체 개수 : %s, 전체 정확도 : %s" % (total_precision, total_num, total_precision / total_num))

    print("작업 종료")