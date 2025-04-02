import cv2
import torch
import time
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO('yolov8n.pt')
model.to(device)

CLASS_NAMES = model.names  

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    results = model(frame)

    detections_list = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() 
        confidences = result.boxes.conf.cpu().numpy()  
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)  
            class_name = CLASS_NAMES.get(class_id, "Unknown")  

            detection = {
                "class": class_name,
                "confidence": round(float(conf), 2),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            }
            detections_list.append(detection)

            label = f"{class_name} | {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 Live Feed", frame)

    print(detections_list)

    if cv2.waitKey(30) == 27: 
        break

cap.release()
cv2.destroyAllWindows()
