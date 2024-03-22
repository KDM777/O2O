from ultralytics import YOLO
import os
import cv2
import json
import numpy as np
import matplotlib.pylab as plt

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8): 
    try: 
        n = np.fromfile(filename, dtype) 
        img = cv2.imdecode(n, flags) 
        return img 
    except Exception as e: 
        print(e) 
        return None
'''
model = YOLO('C:/Users/iialab/Desktop/o2o/v6/runs/detect/train/weights/best.pt')

test_folder='C:/Users/iialab/Desktop/o2o/unnoise_data/train/images/'
'''

if __name__ == '__main__':
    model = YOLO('C:/Users/iialab/Desktop/o2o/o2o_begin/runs/detect/train9/weights/best.pt')  # 저장된 모델인 'best.pt' 로드
    image_folders='C:/Users/iialab/Desktop/o2o/god/test/images/' # 이름을 가져오는 폴더
    name_folder='C:/Users/iialab/Desktop/o2o/paper_ftmatch/name_data' # 이름을 저장하는 폴더
    
    #for image_folder in image_folders:
    image_files = os.listdir(image_folders)
    
    for image_file in image_files:
        image_path = os.path.join(image_folders, image_file)

        # 이미지에 대한 객체 감지 수행
        results = model.predict(image_path)

        # 결과 이미지 그리기
        img = imread(image_path)
        if img is None:
            print(f"오류: 이미지를 찾을 수 없거나 로드할 수 없습니다 - {image_path}")
            continue

        results_json = results[0].tojson()  
        results_dict = json.loads(results_json)
        for pred in results_dict:
            cls=int(pred['class'])

            if(cls == 0):
                x1=int(pred['box']['x1'])
                y1=int(pred['box']['y1'])
                x2=int(pred['box']['x2'])
                y2=int(pred['box']['y2'])
                conf=pred['confidence']
                modify_img=img[y1:y2, x1:x2].copy()

                name_image_path = os.path.join(name_folder, f"{image_file.split('.')[0]}_{x1}_{y1}_{x2}_{y2}.jpg")
                cv2.imwrite(name_image_path, modify_img)
                '''
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f'{cls}: {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                '''
                # 결과 이미지 보여주기
        
        cv2.imshow('Object Detection', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
        