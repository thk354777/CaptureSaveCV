import cv2
import sys
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QWidget,
    QFileDialog,
)

from ultralytics import YOLO
import numpy as np
from PIL import Image

class VideoCaptureThread(QThread):
    new_frame = Signal(QImage)

    def __init__(self):
        super().__init__()
        self.capture = cv2.VideoCapture(0)
        self.running = True
        self.model = YOLO('best2.14.pt')  # Initialize your YOLO model here

    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                continue

            # Perform object detection on the frame
            frame_with_objects = self.detect_objects(frame)

            # Convert OpenCV BGR image to QImage
            height, width, channel = frame_with_objects[0].shape
            bytes_per_line = 3 * width
            q_image = QImage(frame_with_objects[0].data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            self.new_frame.emit(q_image)

    def detect_objects(self, frame):
        
        # Perform object detection using your YOLO model here
        results = self.model.predict(frame, conf=0.6, show=False)
        obj_lists = self.model.names  # Model Classes {0: 'cookie', 1: 'croissant', 2: 'donut'}

        objs = results[0].boxes.numpy()  # Arrays of Predicted result
        obj_count = {value: key for key, value in obj_lists.items()}
        obj_lists_count = dict.fromkeys({value: key for key, value in obj_lists.items()}, 0)

        if objs.shape[0] != 0:  # Check if object > 0 piece.

            for obj in objs:
                detected_obj = obj_lists[int(obj.cls[0])]  # Change Object index to name.
                obj_lists_count[detected_obj] += 1

        # Draw bounding boxes and labels on the frame
        frame_with_objects = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 255, 0)
        thickness = 2
        coor_x = coor_y = 50

        for bread, value in obj_lists_count.items():
            if value > 1:
                text = f'{value} {bread}s'
            else:
                text = f'{value} {bread}'

            coordinates = (coor_x, coor_y)
            frame_with_objects = cv2.putText(frame_with_objects, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
            coor_y += 50

        return frame_with_objects, obj_lists_count

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Capture and Save")
        self.setGeometry(25, 25, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # Align the video to the left and top
        self.layout.addWidget(self.video_label)

        self.capture_button = QPushButton("Capture Image", self)
        self.save_button = QPushButton("Save Image", self)

        self.layout.addWidget(self.capture_button, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.save_button, alignment=Qt.AlignCenter)

        # Add QLabel for displaying the captured image
        self.captured_label = QLabel(self)
        self.layout.addWidget(self.captured_label, alignment=Qt.AlignCenter)

        self.capture_button.clicked.connect(self.capture_image)
        self.save_button.clicked.connect(self.save_image)

        self.video_thread = VideoCaptureThread()
        self.video_thread.new_frame.connect(self.update_video_label)
        self.video_thread.start()

        self.latest_frame = None
        self.captured_frame = None  # Initialize captured_frame

    @Slot(QImage)
    def update_video_label(self, frame):
        self.latest_frame = frame
        pixmap = QPixmap.fromImage(frame)

        # Calculate the size for displaying the video with the desired width
        desired_width = 900  # Change this to your desired width
        scaled_pixmap = pixmap.scaledToWidth(desired_width, Qt.SmoothTransformation)

        self.video_label.setPixmap(scaled_pixmap)

    

    @Slot()
    def capture_image(self):
        if self.latest_frame is not None:
            self.captured_frame = self.latest_frame

            # Convert QImage to QPixmap
            pixmap = QPixmap(self.captured_frame)

            # Convert QPixmap to QImage
            image = pixmap.toImage()

            # Assuming you have a QImage object named 'image'
            width, height = image.width(), image.height()

            # Convert QImage to PIL Image
            pil_image = Image.fromqpixmap(image)  # Convert QImage to PIL Image

            # Convert PIL Image to NumPy array
            numpy_array = np.array(pil_image)

            # If you need RGB format (ignoring alpha channel for transparency)
            rgb_array = numpy_array[:, :, :3]  # Extract RGB channels, ignore the alpha channel
            results = self.video_thread.detect_objects(rgb_array)
            #print(results)
            print(results[1])
            with open("object_counts.txt", "w") as file:
                file.write(str(results[1]) + "\n")

            pixmap = QPixmap.fromImage(self.captured_frame).scaled(32, 24, Qt.KeepAspectRatio)
            self.captured_label.setPixmap(pixmap)
            



      
            
    @Slot()
    def save_image(self):
        if self.captured_frame is not None:
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg);;All Files (*)", options=options)
            if file_path:
                if self.captured_frame.save(file_path):
                    print(f"Image saved as {file_path}")
                    
                else:
                    print("Failed to save image.")

    def closeEvent(self, event):
        self.video_thread.stop()
        self.video_thread.capture.release()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
