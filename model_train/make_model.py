from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('C:/Users/iialab/Desktop/original/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml')  # load a pretrained YOLOv8n detection model
    
    model.train(data='C:/Users/iialab/Desktop/o2o/v7/o2o.yaml', epochs=100, patience=30, batch=1, imgsz=640) 
    


    print(type(model.names), len(model.names))

    print(model.names)

    results = model.predict(source='C:/Users/iialab/Desktop/o2o/v7/test/images/', save=True)
