from ultralytics import YOLO

model = YOLO('yolov8m-seg.pt')

model.info()
results = model.predict(0, show = True, save = True)
