import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Class names for COCO dataset (YOLO default classes)
CLASS_NAMES = model.names  # Dictionary mapping class ID to label

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLOv8 model on the frame
    results = model(frame)

    # Store detections in a dictionary (for HTTP request)
    detections_list = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)  # Convert to integers
            class_name = CLASS_NAMES.get(class_id, "Unknown")  # Get class label

            # Store data in a dictionary for sending via HTTP request
            detection = {
                "class": class_name,
                "confidence": round(float(conf), 2),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            }
            detections_list.append(detection)

            # Draw bounding box and label
            label = f"{class_name} | {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame with detections
    cv2.imshow("YOLOv8 Live Feed", frame)

    # Print detection results (simulate sending via HTTP request)
    print(detections_list)  

    # Exit when ESC key is pressed, delay of 5ms for next frame processing
    if cv2.waitKey(30) == 27:  # ESC key
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
