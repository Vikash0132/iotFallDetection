import sys
import yaml
import cv2

from pathlib import Path
from ultralytics import YOLO

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QMessageBox
)

from vision import VisionEngine


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


class VideoWorker(QThread):

    frame_ready = Signal(QImage)
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, video_path, config):
        super().__init__()

        self.video_path = video_path
        self.config = config

        self.running = True
        self.paused = False
        self.playback_speed = 1.0

    def set_paused(self, value):
        self.paused = value

    def set_speed(self, value):
        self.playback_speed = value

    def stop(self):
        self.running = False

    def run(self):
        capture = cv2.VideoCapture(self.video_path)

        if not capture.isOpened():
            self.error_occurred.emit("Could not open the selected video.")
            return

        try:
            model_path = self.config["model"]["path"]

            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"YOLO model not found: {model_path}"
                )

            self.status_changed.emit("Loading YOLO model...")
            model = YOLO(model_path)

            engine = VisionEngine(self.config, model)

            fps = capture.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                fps = 25

            while self.running:

                if self.paused:
                    self.msleep(100)
                    continue

                success, frame = capture.read()

                if not success:
                    if self.config["video"]["loop"]:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                    break

                try:
                    processed = engine.process(frame)
                except Exception as error:
                    self.error_occurred.emit(
                        f"Video processing error: {error}"
                    )
                    break

                height, width = processed.shape[:2]

                image = QImage(
                    processed.data,
                    width,
                    height,
                    processed.strides[0],
                    QImage.Format.Format_BGR888
                ).copy()

                self.frame_ready.emit(image)

                delay = int(
                    1000 / (fps * max(0.1, self.playback_speed))
                )

                self.msleep(max(1, delay))

            self.status_changed.emit("Playback finished.")

        except Exception as error:
            self.error_occurred.emit(str(error))

        finally:
            capture.release()


class FallDetectionApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("CCTV Fall Detection System")
        self.resize(1300, 850)

        self.config = load_config()
        self.worker = None

        self.video_label = QLabel("Select an MP4 video to begin")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(800, 500)
        self.video_label.setStyleSheet(
            "background-color: #151515; color: white;"
            "font-size: 20px;"
        )

        self.status_label = QLabel("Status: Waiting for video")

        self.open_button = QPushButton("Open Video")
        self.play_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")

        self.speed_combo = QComboBox()
        self.speed_combo.addItems([
            "0.5x",
            "1.0x",
            "1.5x",
            "2.0x"
        ])
        self.speed_combo.setCurrentText("1.0x")

        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        controls = QHBoxLayout()
        controls.addWidget(self.open_button)
        controls.addWidget(self.play_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(QLabel("Speed:"))
        controls.addWidget(self.speed_combo)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label, 1)
        layout.addLayout(controls)
        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

        self.open_button.clicked.connect(self.open_video)
        self.play_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_video)
        self.speed_combo.currentTextChanged.connect(
            self.change_speed
        )

    def open_video(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CCTV Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )

        if not video_path:
            return

        self.stop_video()

        self.worker = VideoWorker(video_path, self.config)

        self.worker.frame_ready.connect(self.update_frame)
        self.worker.status_changed.connect(self.update_status)
        self.worker.error_occurred.connect(self.show_error)

        self.worker.start()

        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.play_button.setText("Pause")

        self.status_label.setText("Status: Starting...")

    def update_frame(self, image):
        pixmap = QPixmap.fromImage(image)

        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.video_label.setPixmap(scaled)

    def toggle_pause(self):
        if self.worker is None:
            return

        paused = not self.worker.paused
        self.worker.set_paused(paused)

        self.play_button.setText(
            "Play" if paused else "Pause"
        )

        self.status_label.setText(
            "Status: Paused" if paused else "Status: Playing"
        )

    def change_speed(self, value):
        if self.worker is None:
            return

        speed = float(value.replace("x", ""))
        self.worker.set_speed(speed)

    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    def show_error(self, message):
        self.status_label.setText("Status: Error")

        QMessageBox.critical(
            self,
            "Fall Detection Error",
            message
        )

    def stop_video(self):
        if self.worker is not None:
            if self.worker.isRunning():
                self.worker.stop()
                self.worker.wait()

            self.worker = None

        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.play_button.setText("Pause")

        self.status_label.setText("Status: Stopped")

    def closeEvent(self, event):
        self.stop_video()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = FallDetectionApp()
    window.show()

    sys.exit(app.exec())