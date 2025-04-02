import sys
import cv2
import torch
import time
import torchvision.transforms as T
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QComboBox, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

def convert_cv_qt(cv_img):
    """Convert from an OpenCV image (BGR) to QImage (RGB)."""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return q_img

class LiveFeedWorker(QThread):
    changePixmap = pyqtSignal(QImage)
    finished = pyqtSignal()

    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type
        self._running = True

    def run(self):
        if self.model_type == "YOLOv8":
            self.run_yolo()
        elif self.model_type == "Faster R-CNN":
            self.run_faster_rcnn()
        self.finished.emit()

    def run_yolo(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO('yolov8n.pt')
        model.to(device)
        CLASS_NAMES = model.names

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        prev_time = time.time()
        while self._running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            results = model(frame)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = CLASS_NAMES.get(class_id, "Unknown")
                    label = f"{class_name} | {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            q_img = convert_cv_qt(frame)
            self.changePixmap.emit(q_img)
            self.msleep(30)

        cap.release()

    def run_faster_rcnn(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
        rcnn_model.to(device)
        rcnn_model.eval()

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
            return transform(image).unsqueeze(0).to(device)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while self._running:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            image_tensor = preprocess_image(frame)
            with torch.no_grad():
                outputs = rcnn_model(image_tensor)

            for box, score, label in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels']):
                if score > 0.5:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    class_id = int(label.item())
                    class_name = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else "Unknown"
                    confidence = round(float(score.item()), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{class_name} | {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            total_time = time.time() - start_time
            fps = 1.0 / total_time if total_time > 0 else 0.0

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            q_img = convert_cv_qt(frame)
            self.changePixmap.emit(q_img)
            self.msleep(30)

        cap.release()

    def stop(self):
        self._running = False

class CompareWorker(QThread):
    changePixmapYOLO = pyqtSignal(QImage)
    changePixmapRCNN = pyqtSignal(QImage)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._running = True

    def run(self):
        # Load models outside the loop
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        yolo_model = YOLO('yolov8n.pt')
        yolo_model.to(device)
        yolo_class_names = yolo_model.names

        rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
        rcnn_model.to(device)
        rcnn_model.eval()
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
            return transform(image).unsqueeze(0).to(device)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while self._running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Create a copy for each model
            frame_yolo = frame.copy()
            frame_rcnn = frame.copy()

            # Process YOLO
            results = yolo_model(frame_yolo)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = yolo_class_names.get(class_id, "Unknown")
                    label = f"{class_name} | {conf:.2f}"
                    cv2.rectangle(frame_yolo, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_yolo, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame_yolo, "YOLOv8", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            q_img_yolo = convert_cv_qt(frame_yolo)
            self.changePixmapYOLO.emit(q_img_yolo)

            # Process Faster R-CNN
            image_tensor = preprocess_image(frame_rcnn)
            with torch.no_grad():
                outputs = rcnn_model(image_tensor)
            for box, score, label in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels']):
                if score > 0.5:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    class_id = int(label.item())
                    class_name = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else "Unknown"
                    confidence = round(float(score.item()), 2)
                    cv2.rectangle(frame_rcnn, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame_rcnn, f"{class_name} | {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame_rcnn, "Faster R-CNN", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            q_img_rcnn = convert_cv_qt(frame_rcnn)
            self.changePixmapRCNN.emit(q_img_rcnn)

            self.msleep(30)

        cap.release()
        self.finished.emit()

    def stop(self):
        self._running = False

class LiveFeedWindow(QMainWindow):
    def __init__(self, model_type, main_window):
        super().__init__()
        self.setWindowTitle(f"Live Feed - {model_type}")
        self.setGeometry(150, 150, 800, 600)
        self.main_window = main_window

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.back_button = QPushButton("Back", self)
        self.back_button.setStyleSheet("padding: 8px; font-size: 16px;")
        self.back_button.clicked.connect(self.back)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.back_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.worker = LiveFeedWorker(model_type)
        self.worker.changePixmap.connect(self.setImage)
        self.worker.start()

    def setImage(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def back(self):
        self.worker.stop()
        self.worker.wait()
        self.close()
        self.main_window.show()

class CompareWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        self.setWindowTitle("Compare: YOLOv8 vs Faster R-CNN")
        self.setGeometry(150, 150, 1200, 600)
        self.main_window = main_window

        # Two video feed labels side by side
        self.label_yolo = QLabel(self)
        self.label_yolo.setAlignment(Qt.AlignCenter)
        self.label_rcnn = QLabel(self)
        self.label_rcnn.setAlignment(Qt.AlignCenter)

        # Back button
        self.back_button = QPushButton("Back", self)
        self.back_button.setStyleSheet("padding: 8px; font-size: 16px;")
        self.back_button.clicked.connect(self.back)

        feeds_layout = QHBoxLayout()
        feeds_layout.addWidget(self.label_yolo)
        feeds_layout.addWidget(self.label_rcnn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(feeds_layout)
        main_layout.addWidget(self.back_button)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.worker = CompareWorker()
        self.worker.changePixmapYOLO.connect(self.setImageYOLO)
        self.worker.changePixmapRCNN.connect(self.setImageRCNN)
        self.worker.start()

    def setImageYOLO(self, image):
        self.label_yolo.setPixmap(QPixmap.fromImage(image))

    def setImageRCNN(self, image):
        self.label_rcnn.setPixmap(QPixmap.fromImage(image))

    def back(self):
        self.worker.stop()
        self.worker.wait()
        self.close()
        self.main_window.show()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Group 3 Project")
        self.setGeometry(100, 100, 500, 300)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel#titleLabel {
                font-size: 28px;
                font-weight: bold;
                color: #333;
            }
            QComboBox, QPushButton {
                font-size: 18px;
                padding: 6px;
            }
            QPushButton {
                background-color: #007ACC;
                color: #fff;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #005F99;
            }
        """)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        self.title_label = QLabel("Group 3 Project", self)
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        # Dropdown for model selection including "Compare"
        self.combo = QComboBox()
        self.combo.addItems(["YOLOv8", "Faster R-CNN", "Compare"])
        layout.addWidget(self.combo)

        self.button = QPushButton("Start")
        self.button.clicked.connect(self.start_model)
        layout.addWidget(self.button)

        self.setCentralWidget(container)

    def start_model(self):
        model_choice = self.combo.currentText()
        self.hide()
        if model_choice == "Compare":
            self.compare_window = CompareWindow(self)
            self.compare_window.show()
        else:
            self.live_feed_window = LiveFeedWindow(model_choice, self)
            self.live_feed_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
