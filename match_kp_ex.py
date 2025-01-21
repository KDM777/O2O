import cv2, numpy as np

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8): 
    try: 
        n = np.fromfile(filename, dtype) 
        img = cv2.imdecode(n, flags) 
        return img 
    except Exception as e: 
        print(e) 
        return None
    
img1 = cv2.imread('C:/Users/iialab/Desktop/o2o/god/demo/images/6.jpg')
img2 = cv2.imread('C:/Users/iialab/Desktop/o2o/paper_ftmatch/name_data_individual/ma.jpg')

img1=cv2.resize(img1, (640,640))
#img2=cv2.resize(img2, (640,640))
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT 서술자 추출기 생성 ---①
detector = cv2.xfeatures2d.SURF_create()
# 각 영상에 대해 키 포인트와 서술자 추출 ---②
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

# BFMatcher 생성, L1 거리, 상호 체크 ---③
matcher = cv2.BFMatcher()
# 매칭 계산 ---④
matches = matcher.knnMatch(desc1, desc2, 2)
        
        # 좋은 매칭 결과 선별
good_matches = [] 
for m in matches: # matches는 두개의 리스트로 구성
    if m[0].distance / m[1].distance <0.7: # 임계점 0.7
        good_matches.append(m[0]) # 매칭 결과 그리기 ---⑤

res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# 결과 출력 
cv2.imwrite('C:/Users/iialab/Desktop/o2o/paper_ftmatch/name_data_img_individual/get.jpg', res)
cv2.waitKey()
cv2.destroyAllWindows()