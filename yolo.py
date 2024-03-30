from ultralytics import YOLO

import cv2
import warnings
warnings.filterwarnings('ignore')


model = YOLO('best.pt')


def get_prediction(image):
    results = model.predict(image, imgsz=640, conf=0.25, iou=0.45)
    results = results[0]
    for i in range(len(results.boxes)):
        box = results.boxes[i]
        tensor = box.xyxy[0]
        x1 = int(tensor[0].item())
        y1 = int(tensor[1].item())
        x2 = int(tensor[2].item())
        y2 = int(tensor[3].item())
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
    return image


vid = cv2.VideoCapture("videos/file1.mp4")

success, frame = vid.read()
while success:
    success, frame = vid.read()
    cv2.imshow('frame', get_prediction(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release()
cv2.destroyAllWindows()
