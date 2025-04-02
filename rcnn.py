import cv2
import torch
import time
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load Faster R-CNN model
rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
rcnn_model.eval()  # Set to evaluation mode

# COCO class labels (for Faster R-CNN, based on COCO dataset)
COCO_LABELS = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe',
    'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table',
    'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def preprocess_image(image):
    transform = T.Compose([T.ToTensor()])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    pre_process_start = time.time()
    image_tensor = preprocess_image(frame)
    pre_process_time = time.time() - pre_process_start
    
    inference_start = time.time()
    with torch.no_grad():
        outputs = rcnn_model(image_tensor)
    inference_time = time.time() - inference_start
    
    post_process_start = time.time()
    detections_list = []
    
    for i, (box, score, label) in enumerate(zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels'])):
        if score > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box.tolist())
            class_id = int(label.item())
            class_name = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else "Unknown"
            confidence = round(float(score.item()), 2)
            
            detection = {"class": class_name, "confidence": confidence, "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
            detections_list.append(detection)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_name} | {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    post_process_time = time.time() - post_process_start
    total_time = time.time() - start_time
    
    cv2.imshow("Faster R-CNN Live Feed", frame)
    print(f"Detections: {detections_list}")
    print(f"Pre-processing Time: {pre_process_time:.4f}s, Inference Time: {inference_time:.4f}s, Post-processing Time: {post_process_time:.4f}s, Total Time: {total_time:.4f}s")
    
    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
