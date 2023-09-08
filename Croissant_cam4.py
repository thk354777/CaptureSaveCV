import cv2
import sys
import os
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

class VideoCaptureThread(QThread):
    new_frame = Signal(QImage)

    def __init__(self):
        super().__init__()
        self.capture = cv2.VideoCapture(0)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                continue

            # Convert OpenCV BGR image to QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            self.new_frame.emit(q_image)

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Capture and Save")
        self.setGeometry(50, 50, 400, 300)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

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
        self.video_label.setPixmap(pixmap)

    @Slot()
    def capture_image(self):
        if self.latest_frame is not None:
            self.captured_frame = self.latest_frame
            # Display the captured image
            pixmap = QPixmap.fromImage(self.captured_frame).scaled(320, 240, Qt.KeepAspectRatio)
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
