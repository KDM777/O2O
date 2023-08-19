from ultralytics import YOLO
import os
import cv2
import json
import numpy as np
import matplotlib.pylab as plt
from PIL import Image, ImageFont, ImageDraw
import json

import cv2
import os

def compare_ftdetect(img, folder_path, using_FAST=True):
    feature_detector = {
        'SIFT': cv2.SIFT_create(),
        
    }

    template_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            img_path = os.path.join(root, file)
            template_paths.append(img_path)

    similarity_scores = {}

    for detector_type, detector in feature_detector.items():
        for template_path in template_paths:
            img2 = imread(template_path)
            img2 = cv2.resize(img2, (640, 640))

            kp1, des1 = detector.compute(img, None)
            kp2, des2 = detector.compute(img2, None)


            matcher = cv2.BFMatcher()
            
            matches = matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:30]
            similarity_scores[template_path] = len(good_matches)

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

def get_mouse_coordinates(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        mid_x=(start_x+end_x)//2
        mid_y=(start_y+end_y)//2
        print(f"시작점 좌표: (x={start_x}, y={start_y})")
        print(f"끝점 좌표: (x={end_x}, y={end_y})")
        print(f"중간 좌표: (x={mid_x}, y={mid_y})")
        if start_x != end_x and start_y != end_y:
            modify_img = img[start_y:end_y, start_x:end_x].copy()
            predictName = compare_ftdetect(modify_img, name_folder)
            print(f"Predicted Name: {predictName}")
            fontpath="C:/Users/iialab/Desktop/o2o/o2o_begin/fonts/gulim.ttc"
            font=ImageFont.truetype(fontpath, 20)
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw=ImageDraw.Draw(img_pil)
            draw.text((start_x, start_y - 10), predictName, font=font, fill=(0, 255, 0,2))
            #draw.text((end_x, end_y - 10), predictName, font=font, fill=(0, 255, 0,2))
            img_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            cv2.imshow('Object Detection', img_with_text)
            cv2.waitKey(0)
            
            #json 파일 저장
            snack=dict()
            snack["mid_x"]=mid_x
            snack["mid_y"]=mid_y
            pred=str(predictName)
            print(pred)
            snack_group["%s"%(pred)]=snack

            #json 파일로 저장
            with open('C:/Users/iialab/Desktop/o2o/v1/db/test.json', 'w', encoding='utf-8') as make_file:
                json.dump(snack_group, make_file, ensure_ascii=False, indent="\t")
            
            # 저장한 파일 출력하기
            with open('C:/Users/iialab/Desktop/o2o/v1/db/test.json', 'r') as f:
                json_data = json.load(f)
            print(json.dumps(json_data, indent="\t") )



    elif event == cv2.EVENT_MOUSEMOVE: # 마우스가 움직일 때 발생
        if drawing:
            temp_img = img.copy()
            cv2.rectangle(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow('Object Detection', temp_img)

if __name__ == '__main__':
    model = YOLO('C:/Users/iialab/runs/detect/train17/weights/best.pt')  # 저장된 모델인 'best.pt' 로드
    test_folder='C:/Users/iialab/Desktop/o2o/shelf/image/'
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
            '''
            modify_img=img[y1:y2, x1:x2].copy()        
            input_hsv = cv2.cvtColor(modify_img, cv2.COLOR_BGR2HSV)
            input_hist = cv2.calcHist([input_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

            predictName = compare_histograms(input_hist, name_folder, image_file)
            '''
            modify_img=img[y1:y2, x1:x2].copy()        
            
            predictName = compare_ftdetect(modify_img, name_folder)
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
        
        cv2.imshow('Object Detection', img)
        cv2.setMouseCallback('Object Detection', get_mouse_coordinates)
        drawing = False
        start_x, start_y, end_x, end_y = -1, -1, -1, -1

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC 키를 누르면 종료
                break
        
        cv2.waitKey(0)

    cv2.destroyAllWindows()    
    
