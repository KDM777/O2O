from ultralytics import YOLO
import os
import cv2
import json
import numpy as np
import matplotlib.pylab as plt
from PIL import Image, ImageFont, ImageDraw

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8): 
    try: 
        n = np.fromfile(filename, dtype) 
        img = cv2.imdecode(n, flags) 
        return img 
    except Exception as e: 
        print(e) 
        return None
#파일 이름을 해서 비교할 때
'''
def compare_histograms(query_hist, folder_path,inputImageName):
    imgs = os.listdir(folder_path)
    

    hists = []
    similarity_scores_bha = []  # List to store similarity scores for each image
    similarity_scores_int = []
    similarity_scores_cor = []
    imgs.sort()
    #print(imgs)
    for img_file in imgs:
        img_path = os.path.join(folder_path, img_file)
        img = imread(img_path)

        # BGR 이미지를 HSV 이미지로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 히스토그램 연산(파라미터 순서 : 이미지, 채널, Mask, 크기, 범위)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # 정규화(파라미터 순서 : 정규화 전 데이터, 정규화 후 데이터, 시작 범위, 끝 범위, 정규화 알고리즘)
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        # hists 리스트에 저장
        hists.append(hist)

        # 입력된 이미지와 폴더 내 이미지들 간의 BHATTACHARYYA 히스토그램 비교 수행
        ret = cv2.compareHist(query_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        similarity_scores_bha.append(1-ret)

        ret = cv2.compareHist(query_hist, hist, cv2.HISTCMP_INTERSECT )
        similarity_scores_int.append(ret)

        ret = cv2.compareHist(query_hist, hist, cv2.HISTCMP_CORREL )
        similarity_scores_cor.append(ret)

    voting = [0 for _ in range(int(523))] # 과자이름 최대 개수 입력 칸

    for sol in [similarity_scores_bha,similarity_scores_int,similarity_scores_cor]:
        max_score_idx = np.argmax(sol)
        voting[max_score_idx] += 1

    votidx = voting.index(max(voting))    
    most_similar_image = imgs[votidx]
    predictName = most_similar_image.split(" ")[0]    
    print(f"입력 이미지 이름: {inputImageName}, name_folder에 가장 유사도 높은 이미지 이름: {predictName}")

    return predictName
'''
#폴더 이름을 해서 비교할 때
def compare_histograms(query_hist, folder_path, inputImageName):
    # 이미지 파일들을 재귀적으로 탐색하여 저장할 리스트
    imgs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            img_path = os.path.join(root, file)
            imgs.append(img_path)

    hists = []
    similarity_scores = {}  # 딕셔너리로 유사도 저장

    for img_path in imgs:
        img = imread(img_path)

        # BGR 이미지를 HSV 이미지로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 히스토그램 연산(파라미터 순서 : 이미지, 채널, Mask, 크기, 범위)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # 정규화(파라미터 순서 : 정규화 전 데이터, 정규화 후 데이터, 시작 범위, 끝 범위, 정규화 알고리즘)
        #cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        # hists 리스트에 저장
        hists.append(hist)

        # 입력된 이미지와 폴더 내 이미지들 간의 히스토그램 비교 수행
        ret = cv2.compareHist(query_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        similarity_scores[img_path] = ret

    # 유사도가 가장 높은 폴더 경로를 찾음
    most_similar_folder = min(similarity_scores, key=similarity_scores.get)
    # 폴더 이름 추출 (폴더 경로에서 마지막 폴더 이름만 추출)
    predictName = os.path.basename(os.path.dirname(most_similar_folder))
    #print(f"입력 이미지 이름: {inputImageName}, name_folder에 가장 유사도 높은 폴더 이름: {predictName}")

    return predictName

def get_mouse_coordinates(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y

        print(f"시작점 좌표: (x={start_x}, y={start_y})")
        print(f"끝점 좌표: (x={end_x}, y={end_y})")

if __name__ == '__main__':
    model = YOLO('C:/Users/iialab/runs/detect/train17/weights/best.pt')  # 저장된 모델인 'best.pt' 로드
    test_folder='C:/Users/iialab/Desktop/o2o/shelf/image/'
    # test_folder = 'C:/Users/iialab/Desktop/o2o/v1/test/images/'  # 테스트 이미지 폴더 경로
    name_folder='C:/Users/iialab/Desktop/o2o/v1/name/' #이름만 모아 놓은 곳
    
    
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
    #0 : name 1 : snack
    #진열대는 반대
        for pred in results_dict:
            cls = int(pred['class'])
            conf = pred['confidence']
            if cls == 0:
                x1 = int(pred['box']['x1'])
                y1 = int(pred['box']['y1'])
                x2 = int(pred['box']['x2'])
                y2 = int(pred['box']['y2'])
                snack_dict[str(num)]=[(x1, y1), (x2, y2),conf]
                num += 1
            
        print(snack_dict)

        for pred in results_dict:
            cls = int(pred['class'])
            conf = pred['confidence']
                
            if cls == 1:
                x1 = int(pred['box']['x1'])
                y1 = int(pred['box']['y1'])
                x2 = int(pred['box']['x2'])
                y2 = int(pred['box']['y2'])
                for snack in snack_dict.keys():
                    left = snack_dict[snack][0][0]
                    top = snack_dict[snack][0][1]
                    right = snack_dict[snack][1][0]
                    bottom = snack_dict[snack][1][1]
                
                    if (x1>=left and y1>=top) and (x2<=right and y2<= bottom) : 
                        if snack in name_dict:
                            if name_dict[snack][2] < conf:
                                name_dict[snack] = [(x1, y1), (x2, y2), conf]
                        else:
                            name_dict[snack] = [(x1, y1), (x2, y2), conf]
                

        print(name_dict)

        for name in name_dict.keys():
            x1 = name_dict[name][0][0]
            y1 = name_dict[name][0][1]
            x2 = name_dict[name][1][0]
            y2 = name_dict[name][1][1]
            confname = name_dict[name][2]
            left = snack_dict[name][0][0]
            top = snack_dict[name][0][1]
            right = snack_dict[name][1][0]
            bottom = snack_dict[name][1][1]
            confsnack = snack_dict[name][2]

            modify_img=img[y1:y2, x1:x2].copy()        
            input_hsv = cv2.cvtColor(modify_img, cv2.COLOR_BGR2HSV)
            input_hist = cv2.calcHist([input_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

            predictName = compare_histograms(input_hist, name_folder, image_file)
            
            fontpath="C:/Users/iialab/Desktop/o2o/o2o_begin/fonts/gulim.ttc"
            font=ImageFont.truetype(fontpath, 20)
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw=ImageDraw.Draw(img_pil)
            draw.text((x1, y1 - 10), predictName, font=font, fill=(0, 255, 0,2))

            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            cls = "name"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(img, f'{cls}: {confname:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) 
            

            cls = "snack"
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(img, f'{cls}: {confsnack:.2f}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) 
            #cv2.putText(img, f'{predictName}', (x1, int((y1+y2)/2)), font, 0.9, (255, 255, 255), 2) 
        #_1split

        cv2.imshow('Object Detection', img)
        cv2.setMouseCallback('Object Detection', get_mouse_coordinates)
        drawing = False
        start_x, start_y, end_x, end_y = -1, -1, -1, -1

        
        if drawing:
            temp_img = img.copy()
            cv2.rectangle(temp_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            cv2.imshow('Object Detection', temp_img)
            modify_img=img[start_y:end_y, start_x:end_x].copy()        
            input_hsv = cv2.cvtColor(modify_img, cv2.COLOR_BGR2HSV)
            input_hist = cv2.calcHist([input_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

            predictName = compare_histograms(input_hist, name_folder, image_file)
            
        else:
            cv2.imshow('Object Detection', img)
        
        '''
        while True:
        # 이미지에 네모를 그릴 때
            if drawing:
                temp_img = img.copy()
                cv2.rectangle(temp_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                cv2.imshow('Object Detection', temp_img)
            else:
                cv2.imshow('Object Detection', img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC 키를 누르면 종료
                break
        
        cv2.waitKey(0)
        '''
    cv2.destroyAllWindows()    
    
    