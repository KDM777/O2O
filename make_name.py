import os

if __name__ == '__main__':
    img_folder = 'C:/Users/iialab/Desktop/o2o/paper_ftmatch/name_data/snack'
    #label_folder='C:/Users/iialab/Desktop/o2o/god/test/labels/'
    #name_folder = 'C:/Users/iialab/Desktop/o2o/paper_ftmatch/name_data/'  # 이름만 모아 놓은 곳
    #name_folder = 'C:/Users/iialab/Desktop/o2o/total_name_2/'  # 이름만 모아 놓은 곳
    #name_folder = 'C:/Users/iialab/Desktop/o2o/total_name_3/'  # 이름만 모아 놓은 곳
    f=open("name2.txt","w+")
    image_files = os.listdir(img_folder)
    #label_files = os.listdir(label_folder)

    for image_file in image_files:
        #image_path = os.path.join(img_folder, image_file)
        
        f.write(image_file+'\n')
    f.close