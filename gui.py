# gui.py
# Minimal PyQt5 GUI that integrates with the existing DeepFace analysis logic.
# Drop this file next to main.py and run it. Install PyQt5: pip install PyQt5
#
# Notes:
# - This uses a QThread to read camera frames and perform periodic analysis in that thread.
# - For production, you should reuse your Database/DBLogger and known-embeddings loader.
# - The analysis interval and frame resizing are configurable.

import sys
import time
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from deepface import DeepFace
import statistics
from collections import deque
import os

# Configure these to match your existing project
ANALYZE_INTERVAL = 0.25    # seconds between analyses
SMALL_FRAME_SIZE = (160, 120)
MIN_CONFIDENCE = 60.0
EMOJI_DIR = "."            # emoji images live in repo root or change path

# Load emojis (same approach as main.py)
emotion_images = {
    "happy": "happy.png",
    "surprise": "surprise.png",
    "disgust": "disgust.png",
    "sad": "sad.png",
    "angry": "angry.png",
    "neutral": "neutral.png",
    "fear": "fear.png"
}
EMOJIS = {}
for k, p in emotion_images.items():
    full = os.path.join(EMOJI_DIR, p)
    EMOJIS[k] = cv2.imread(full, cv2.IMREAD_UNCHANGED) if os.path.exists(full) else None

def cvimg_to_qimage(cv_img):
    """Convert BGR OpenCV image to QImage"""
    h, w = cv_img.shape[:2]
    if cv_img.ndim == 2:
        fmt = QtGui.QImage.Format_Grayscale8
        qimg = QtGui.QImage(cv_img.data, w, h, cv_img.strides[0], fmt)
        return qimg.copy()
    if cv_img.shape[2] == 3:
        # BGR -> RGB
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
        return qimg.copy()
    if cv_img.shape[2] == 4:
        rgba = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
        qimg = QtGui.QImage(rgba.data, w, h, rgba.strides[0], QtGui.QImage.Format_RGBA8888)
        return qimg.copy()
    return None

class CameraWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object)            # emits numpy array (BGR)
    analysis_result = QtCore.pyqtSignal(str, float)    # emits (emotion, confidence)

    def __init__(self, src=0, parent=None):
        super().__init__(parent)
        self.src = src
        self.running = True
        self.cap = None
        self.last_analysis = 0
        self.emotion_history = deque(maxlen=8)
        self.last_emotion = "Detecting..."
        self.detector_backend = "opencv"
        # If you have known_embeddings loaded, you can attach them here:
        self.known_embeddings = None  # set externally to reuse precomputed embeddings

    def run(self):
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            print("Failed to open camera")
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            # Emit latest frame (fast)
            self.frame_ready.emit(frame)

            # Throttle analysis
            now = time.time()
            if now - self.last_analysis >= ANALYZE_INTERVAL:
                self.last_analysis = now
                # Resize for speed
                small = cv2.resize(frame, SMALL_FRAME_SIZE)
                try:
                    result = DeepFace.analyze(small, actions=["emotion"], enforce_detection=False, detector_backend=self.detector_backend)
                    if isinstance(result, list):
                        result = result[0]
                    emotions = result.get("emotion", {}) or {}
                    dominant = result.get("dominant_emotion", None)
                    confidence = float(max(emotions.values())) if emotions else 0.0

                    if dominant and confidence >= MIN_CONFIDENCE:
                        self.emotion_history.append(dominant)
                        try:
                            stable = statistics.mode(self.emotion_history)
                        except statistics.StatisticsError:
                            stable = dominant

                        # TODO: plug recognition here if you precomputed embeddings:
                        # rep = DeepFace.represent(small, model_name="Facenet", enforce_detection=False, detector_backend=self.detector_backend)
                        # person = recognize_embedding(rep, self.known_embeddings)  # optional

                        # Only emit when new stable emotion appears (optional)
                        if stable != self.last_emotion:
                            self.last_emotion = stable
                            self.analysis_result.emit(stable, confidence)
                except Exception as e:
                    # keep GUI responsive; optionally log the error
                    # print("Analysis error:", e)
                    pass

            # tiny sleep to avoid hogging CPU when capture is very fast
            time.sleep(0.001)

        # clean up
        try:
            self.cap.release()
        except Exception:
            pass

    def stop(self):
        self.running = False
        self.wait(2000)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MoodMirror GUI")
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background: black")
        self.emoji_label = QtWidgets.QLabel()
        self.emoji_label.setFixedSize(320, 240)
        self.info_label = QtWidgets.QLabel("Person: Unknown\nEmotion: Detecting...")
        self.info_label.setAlignment(QtCore.Qt.AlignCenter)

        start_btn = QtWidgets.QPushButton("Start")
        stop_btn = QtWidgets.QPushButton("Stop")
        start_btn.clicked.connect(self.start)
        stop_btn.clicked.connect(self.stop)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(start_btn)
        btn_layout.addWidget(stop_btn)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.info_label)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.emoji_label)
        h.addLayout(btn_layout)
        layout.addLayout(h)
        self.setLayout(layout)

        # Camera worker
        self.worker = CameraWorker(src=0)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.analysis_result.connect(self.update_emotion)

    @QtCore.pyqtSlot(object)
    def update_frame(self, frame):
        """Display camera frame in QLabel."""
        qimg = cvimg_to_qimage(frame)
        if qimg is not None:
            pix = QtGui.QPixmap.fromImage(qimg).scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio)
            self.video_label.setPixmap(pix)

    @QtCore.pyqtSlot(str, float)
    def update_emotion(self, emotion, confidence):
        """Update textual info and emoji."""
        # update label
        self.info_label.setText(f"Person: Unknown\nEmotion: {emotion} ({confidence:.0f}%)")
        # update emoji if available
        emoji_cv = EMOJIS.get(emotion, None)
        if emoji_cv is not None:
            # resize emoji to fit emoji_label
            emoji_rgb = cv2.cvtColor(emoji_cv, cv2.COLOR_BGRA2RGBA) if emoji_cv.shape[2] == 4 else cv2.cvtColor(emoji_cv, cv2.COLOR_BGR2RGB)
            q = cvimg_to_qimage(emoji_rgb)
            pix = QtGui.QPixmap.fromImage(q).scaled(self.emoji_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.emoji_label.setPixmap(pix)
        else:
            self.emoji_label.clear()

    def start(self):
        if not self.worker.isRunning():
            self.worker = CameraWorker(src=0)
            self.worker.frame_ready.connect(self.update_frame)
            self.worker.analysis_result.connect(self.update_emotion)
            # Optionally attach known_embeddings loader here:
            # self.worker.known_embeddings = load_known_embeddings("faces", model_name="Facenet", detector_backend="opencv")
            self.worker.start()

    def stop(self):
        if self.worker.isRunning():
            self.worker.stop()
            self.video_label.clear()
            self.emoji_label.clear()
            self.info_label.setText("Person: Unknown\nEmotion: Stopped")

    def closeEvent(self, event):
        self.stop()
        event.accept()

def run():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run()
