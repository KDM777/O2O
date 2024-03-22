import os
import cv2
import numpy as np

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8): 
    try: 
        n = np.fromfile(filename, dtype) 
        img = cv2.imdecode(n, flags) 
        return img 
    except Exception as e: 
        print(e) 
        return None

def extract_features(image):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    return kp, des

def match_features(des1, des2):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, 2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

def compare_images(image, folder_path):
    template_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]

    similarity_scores = {}
    for template_path in template_paths:
        template_image = imread(template_path)
        _, template_des = extract_features(template_image)
        _, image_des = extract_features(image)
        matches = match_features(image_des, template_des)
        similarity_scores[template_path] = sum(match.distance for match in matches)
    most_similar_template = max(similarity_scores, key=similarity_scores.get)
    return os.path.basename(os.path.dirname(most_similar_template))

def evaluate_accuracy(img_folder, label_folder, name_folder):
    classes_dict = {}
    with open(os.path.join(label_folder, "classes.txt"), "r") as f:
        for i, line in enumerate(f):
            classes_dict[line.strip()] = i
    
    total_num = 0
    total_precision = 0
    
    for i, (image_file, label_file) in enumerate(zip(os.listdir(img_folder), os.listdir(label_folder))):
        image_path = os.path.join(img_folder, image_file)
        label_path = os.path.join(label_folder, label_file)

        img = imread(image_path)
        if img is None:
            print(f"오류: 이미지를 찾을 수 없거나 로드할 수 없습니다 - {image_path}")
            continue
        
        data_list = []
        with open(label_path, 'r') as file:
            for line in file:
                elements = list(map(float, line.strip().split()))
                data_list.append(elements)
        
        num = 0
        precision = 0

        for pred in data_list:
            left, top, right, bottom, label = list(map(int, pred))
            roi = img[top:bottom, left:right].copy()
            predicted_label = compare_images(roi, name_folder)
            if classes_dict.get(predicted_label) == label:
                precision += 1
            num += 1
        
        total_num += num
        total_precision += precision
        print(f"이미지 {i+1} - 맞는 개수: {precision}, 개수: {num}, 정확도: {precision/num}")
    
    print(f"전체 맞는 개수: {total_precision}, 전체 개수: {total_num}, 전체 정확도: {total_precision/total_num}")

if __name__ == '__main__':
    img_folder = 'C:/Users/iialab/Desktop/o2o/god/demo/images/'
    label_folder = 'C:/Users/iialab/Desktop/o2o/god/demo/labels/'
    ramen_name_folder = 'C:/Users/iialab/Desktop/o2o/paper_ftmatch/name_data/ramen/'  
    snack_name_folder = 'C:/Users/iialab/Desktop/o2o/paper_ftmatch/name_data/snack/'  

    evaluate_accuracy(img_folder, label_folder, ramen_name_folder)