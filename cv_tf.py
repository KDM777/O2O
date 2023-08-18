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

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import os

# TensorFlow Hub 모델을 불러옵니다. 이 예제에서는 Inception ResNet V2를 사용합니다.
embed = hub.load("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5")

# 이미지를 로드하고 전처리하는 함수를 정의합니다.
def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (640, 640))  # Inception ResNet V2의 입력 크기에 맞게 조정
    return image

# 이미지를 임베딩 벡터로 변환하는 함수를 정의합니다.
def get_image_embedding(image):
    image = tf.expand_dims(image, axis=0)  # 배치 차원 추가
    embeddings = embed(image)
    return np.array(embeddings).flatten()

def compare_tf(given_image, image_parent_folder):
    cropped_embedding=get_image_embedding(given_image)

    # 하위 폴더들을 순회하며 유사도를 계산하고, 가장 유사한 폴더를 찾습니다.
    best_similarity = 0.0
    best_folder_name = None

    for subfolder_name in os.listdir(image_parent_folder):
        subfolder_path = os.path.join(image_parent_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            similarities = []

            for image_file in os.listdir(subfolder_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(subfolder_path, image_file)
                    image = load_and_preprocess_image(image_path)
                    image_embedding = get_image_embedding(image)

                    similarity = np.dot(cropped_embedding, image_embedding)  # 임베딩 벡터 간의 내적 계산
                    #cropped_embedding은 get_image_embedding을 거친 주어진 사진
                    similarities.append(similarity)

            average_similarity = np.mean(similarities)
            if average_similarity > best_similarity:
                best_similarity = average_similarity
                best_folder_name = subfolder_name

    if best_folder_name:
        print(f"The most similar folder is: {best_folder_name} with average similarity: {best_similarity}")
    else:
        print("No similar folder found.")

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
            predictName = compare_tf(modify_img, name_folder)
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
                json.dump(snack_group, make_file, ensure_ascii=True, indent="\t") # ensure_ascii를 False로하면(에러뜸) 한글 True로 하면 이상한 단어(에러 안뜸)
            
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
            modify_img=img[y1:y2, x1:x2].copy()        
            
            predictName = compare_tf(modify_img, name_folder)
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
    
