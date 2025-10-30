# -*- coding: ascii -*-
"""
Updated main.py - PyQt5 GUI front-end integrated with DeepFace analysis.

Notes:
- This file uses only ASCII characters.
- To run: pip install opencv-python deepface PyQt5 mysql-connector-python
- Run: python main.py
"""
import sys
import time
import os
import queue
import threading
import logging
from collections import deque

import cv2
import numpy as np
import statistics
from deepface import DeepFace

from PyQt5 import QtCore, QtGui, QtWidgets

from database import Database

logging.getLogger("deepface").setLevel(logging.ERROR)
logging.basicConfig(filename="deepface_log.txt", level=logging.ERROR)

# Config
ANALYZE_INTERVAL = 0.25       # seconds between analyses
SMALL_FRAME_SIZE = (160, 120) # speed/accuracy tradeoff
MIN_CONFIDENCE = 60.0         # minimum emotion confidence to accept
DETECTOR_BACKEND = "opencv"   # faster on CPU
KNOWN_MODEL = "Facenet"       # model used for embeddings
EMOJI_DIR = "."               # where emoji images live
EMOTION_IMAGES = {
    "happy": "happy.png",
    "surprise": "surprise.png",
    "disgust": "disgust.png",
    "sad": "sad.png",
    "angry": "angry.png",
    "neutral": "neutral.png",
    "fear": "fear.png"
}

# Helpers
def l2_normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def load_emojis():
    emojis = {}
    for k, p in EMOTION_IMAGES.items():
        full = os.path.join(EMOJI_DIR, p)
        if os.path.exists(full):
            emojis[k] = cv2.imread(full, cv2.IMREAD_UNCHANGED)
        else:
            emojis[k] = None
    return emojis

EMOJIS = load_emojis()

def overlay_emoji(base_img, emoji):
    if emoji is None:
        return base_img
    try:
        emoji_resized = cv2.resize(emoji, (base_img.shape[1], base_img.shape[0]))
    except Exception:
        return base_img
    if emoji_resized.ndim == 2:
        return base_img
    eh, ew, ec = emoji_resized.shape
    bg = base_img.copy().astype(np.float32)
    if ec == 4:
        alpha = emoji_resized[:, :, 3] / 255.0
        for c in range(3):
            bg[:, :, c] = bg[:, :, c] * (1 - alpha) + emoji_resized[:, :, c] * alpha
        return bg.astype(np.uint8)
    else:
        return emoji_resized

def cvimg_to_qimage(cv_img):
    if cv_img is None:
        return None
    h, w = cv_img.shape[:2]
    if cv_img.ndim == 2:
        fmt = QtGui.QImage.Format_Grayscale8
        qimg = QtGui.QImage(cv_img.data, w, h, cv_img.strides[0], fmt)
        return qimg.copy()
    if cv_img.shape[2] == 3:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
        return qimg.copy()
    if cv_img.shape[2] == 4:
        rgba = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
        qimg = QtGui.QImage(rgba.data, w, h, rgba.strides[0], QtGui.QImage.Format_RGBA8888)
        return qimg.copy()
    return None

# Known embeddings loader
def load_known_embeddings(faces_dir="faces", model_name=KNOWN_MODEL, detector_backend=DETECTOR_BACKEND):
    known = {}
    if not os.path.exists(faces_dir):
        return known
    for person in os.listdir(faces_dir):
        person_path = os.path.join(faces_dir, person)
        if not os.path.isdir(person_path):
            continue
        emb_list = []
        for fname in os.listdir(person_path):
            fpath = os.path.join(person_path, fname)
            try:
                rep = DeepFace.represent(fpath, model_name=model_name, enforce_detection=False, detector_backend=detector_backend)
                if isinstance(rep, list) and len(rep) > 0 and isinstance(rep[0], (list, np.ndarray)):
                    vec = np.array(rep[0], dtype=np.float32)
                else:
                    vec = np.array(rep, dtype=np.float32)
                vec = l2_normalize(vec)
                emb_list.append(vec)
            except Exception as e:
                logging.debug("Failed embedding for %s: %s", fpath, e)
        if emb_list:
            known[person] = emb_list
    return known

def recognize_embedding(embedding, known_embeddings, similarity_threshold=0.60):
    if embedding is None or not known_embeddings:
        return None
    emb = l2_normalize(np.array(embedding))
    best_person = None
    best_score = -1.0
    for person, embs in known_embeddings.items():
        for k_emb in embs:
            score = cosine_similarity(emb, k_emb)
            if score > best_score:
                best_score = score
                best_person = person
    if best_score >= similarity_threshold:
        return best_person
    return None

# DB Logger (threaded)
class DBLogger(threading.Thread):
    def __init__(self, db: Database, flush_timeout=0.5):
        super().__init__(daemon=True)
        self.db = db
        self.q = queue.Queue()
        self.running = True
        self.flush_timeout = flush_timeout

    def enqueue_face_mood(self, person, emotion, confidence=0.0, duration=0.0, timestamp=None):
        self.q.put(("face_mood", (person, emotion, confidence, duration, timestamp)))

    def run(self):
        while self.running:
            try:
                kind, args = self.q.get(timeout=self.flush_timeout)
            except queue.Empty:
                continue
            try:
                if kind == "face_mood":
                    try:
                        self.db.log_face_mood(*args)
                    except TypeError:
                        self.db.log_face_mood(args[0], args[1], args[2], args[3])
            except Exception as e:
                logging.error("DB logging failed: %s", e)

    def stop(self):
        self.running = False

# Camera & Analysis worker (QThread)
class CameraWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object)
    emotion_detected = QtCore.pyqtSignal(str, float, str)

    def __init__(self, src=0, known_embeddings=None, db_logger=None, parent=None):
        super().__init__(parent)
        self.src = src
        self.running = True
        self.known_embeddings = known_embeddings or {}
        self.db_logger = db_logger
        self.last_analysis = 0
        self.emotion_history = deque(maxlen=8)
        self.last_emotion = "Detecting..."
        self.detector_backend = DETECTOR_BACKEND
        self.model = KNOWN_MODEL

    def run(self):
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            print("Failed to open camera")
            return
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                frame = cv2.flip(frame, 1)
                self.frame_ready.emit(frame)

                now = time.time()
                if now - self.last_analysis >= ANALYZE_INTERVAL:
                    self.last_analysis = now
                    try:
                        small = cv2.resize(frame, SMALL_FRAME_SIZE)
                    except Exception:
                        small = frame.copy()
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

                            person_name = "Unknown"
                            try:
                                rep = DeepFace.represent(small, model_name=self.model, enforce_detection=False, detector_backend=self.detector_backend)
                                if isinstance(rep, list) and len(rep) > 0 and isinstance(rep[0], (list, np.ndarray)):
                                    embedding = np.array(rep[0], dtype=np.float32)
                                else:
                                    embedding = np.array(rep, dtype=np.float32)
                                p = recognize_embedding(embedding, self.known_embeddings)
                                if p:
                                    person_name = p
                            except Exception:
                                pass

                            if stable != self.last_emotion:
                                self.last_emotion = stable
                                if self.db_logger:
                                    try:
                                        self.db_logger.enqueue_face_mood(person_name, stable, confidence, 0)
                                    except Exception:
                                        pass
                            self.emotion_detected.emit(stable, confidence, person_name)
                    except Exception as e:
                        logging.debug("Analysis error: %s", e)
                time.sleep(0.001)
        finally:
            try:
                cap.release()
            except Exception:
                pass

    def stop(self):
        self.running = False
        self.wait(2000)

# Main Window
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MoodMirror - GUI")
        self.resize(900, 700)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background: black;")

        self.emoji_label = QtWidgets.QLabel()
        self.emoji_label.setFixedSize(320, 240)
        self.emoji_label.setStyleSheet("background: white;")

        self.info_label = QtWidgets.QLabel("Person: Unknown\nEmotion: Detecting...")
        self.info_label.setAlignment(QtCore.Qt.AlignCenter)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.quit_btn = QtWidgets.QPushButton("Quit")

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.quit_btn)

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self.video_label)
        left_layout.addWidget(self.info_label)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.emoji_label)
        right_layout.addLayout(btn_layout)
        right_layout.addStretch(1)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

        # DB + logger
        self.db = Database(host="localhost", port=1025, user="root", password="root", database="emoji_tracker")
        try:
            self.db.connect()
        except Exception:
            pass
        self.db_logger = DBLogger(self.db)
        self.db_logger.start()

        # Known embeddings
        self.known_embeddings = load_known_embeddings("faces", model_name=KNOWN_MODEL, detector_backend=DETECTOR_BACKEND)

        self.worker = None

        self.start_btn.clicked.connect(self.start_worker)
        self.stop_btn.clicked.connect(self.stop_worker)
        self.quit_btn.clicked.connect(self.close)

        self.current_person = "Unknown"
        self.current_emotion = "Detecting..."
        self.current_confidence = 0.0

        self.blank_display = np.ones((240, 320, 3), dtype=np.uint8) * 255

    @QtCore.pyqtSlot(object)
    def _on_frame(self, frame):
        qimg = cvimg_to_qimage(frame)
        if qimg:
            pix = QtGui.QPixmap.fromImage(qimg).scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.video_label.setPixmap(pix)

    @QtCore.pyqtSlot(str, float, str)
    def _on_emotion(self, emotion, confidence, person):
        self.current_person = person or "Unknown"
        self.current_emotion = emotion or "Detecting..."
        self.current_confidence = confidence or 0.0

        self.info_label.setText("Person: %s\nEmotion: %s (%.0f%%)" % (self.current_person, self.current_emotion, self.current_confidence))

        emoji_cv = EMOJIS.get(self.current_emotion, None)
        if emoji_cv is not None:
            try:
                emoji_rgb = cv2.cvtColor(emoji_cv, cv2.COLOR_BGRA2RGBA) if emoji_cv.shape[2] == 4 else cv2.cvtColor(emoji_cv, cv2.COLOR_BGR2RGB)
                q = cvimg_to_qimage(emoji_rgb)
                pix = QtGui.QPixmap.fromImage(q).scaled(self.emoji_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                self.emoji_label.setPixmap(pix)
            except Exception:
                self.emoji_label.clear()
        else:
            self.emoji_label.clear()

    def start_worker(self):
        if self.worker and self.worker.isRunning():
            return
        self.worker = CameraWorker(src=0, known_embeddings=self.known_embeddings, db_logger=self.db_logger)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.emotion_detected.connect(self._on_emotion)
        self.worker.start()
        self.info_label.setText("Person: Unknown\nEmotion: Starting...")

    def stop_worker(self):
        if self.worker:
            try:
                self.worker.stop()
            except Exception:
                pass
            self.worker = None
        self.video_label.clear()
        self.emoji_label.clear()
        self.info_label.setText("Person: Unknown\nEmotion: Stopped")

    def closeEvent(self, event):
        try:
            self.stop_worker()
        except Exception:
            pass
        try:
            if self.db_logger:
                self.db_logger.stop()
        except Exception:
            pass
        try:
            time.sleep(0.1)
        except Exception:
            pass
        try:
            self.db.close()
        except Exception:
            pass
        event.accept()

# Entry point
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()