# -*- coding: ascii -*-

import sys
import time
import os
import queue
import threading
import logging
import random
import json
import webbrowser
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import statistics
from deepface import DeepFace

from PyQt5 import QtCore, QtGui, QtWidgets

from database import Database

# Try to import pyttsx3 (offline TTS). If not available, TTS becomes a no-op.
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# Try to import Flask for HTTP server
try:
    from flask import Flask, Response, jsonify, stream_with_context, request
except Exception:
    Flask = None

# Logging setup
logger = logging.getLogger("moodmirror")
logger.setLevel(logging.DEBUG)
logging.getLogger("deepface").setLevel(logging.ERROR)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ENABLE_HTTP = True
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 5000

ANALYZE_INTERVAL = 0.25       # seconds between analyses
SMALL_FRAME_SIZE = (160, 120) # quick detection size
ANALYSIS_SIZE = (320, 240)    # size used for DeepFace (better accuracy)
MIN_CONFIDENCE = 60.0
DETECTOR_BACKEND = "opencv"
KNOWN_MODEL = "Facenet"
EMOJI_DIR = "."
EMOTION_IMAGES = {
    "happy": "happy.png",
    "surprise": "surprise.png",
    "disgust": "disgust.png",
    "sad": "sad.png",
    "angry": "angry.png",
    "neutral": "neutral.png",
    "fear": "fear.png"
}

ENABLE_TTS = True
TTS_SPEAK_INTERVAL = 5.0

PHRASES = {
    "happy": ["You look very happy today.", "Nice to see you smiling.", "What a great smile you have today."],
    "surprise": ["You seem surprised. Everything ok?", "Wow, you look surprised!"],
    "disgust": ["You look a bit displeased. Is everything all right?"],
    "sad": ["You look a little sad. I am here if you need me."],
    "angry": ["You look upset. Remember to breathe."],
    "neutral": ["You look calm and neutral."],
    "fear": ["You look concerned. Are you okay?"],
    "no_face": ["No face detected. Please look at the camera."]
}

# -----------------------------------------------------------------------------
# Integration API for in-process dashboards
# -----------------------------------------------------------------------------
_update_listeners = []
_update_listeners_lock = threading.Lock()

def register_update_callback(cb):
    """
    Register a callback to receive payloads from the running app.
    Callback signature: cb(payload_dict)
    """
    if not callable(cb):
        raise ValueError("callback must be callable")
    with _update_listeners_lock:
        if cb not in _update_listeners:
            _update_listeners.append(cb)

def unregister_update_callback(cb):
    """Unregister a previously registered callback."""
    with _update_listeners_lock:
        try:
            _update_listeners.remove(cb)
        except ValueError:
            pass

def _notify_listeners(payload):
    """
    Notify registered in-process listeners with the given payload.
    Each callback is invoked on a daemon thread so listeners cannot block the main loop.
    """
    with _update_listeners_lock:
        listeners = list(_update_listeners)
    for cb in listeners:
        try:
            threading.Thread(target=cb, args=(payload,), daemon=True).start()
        except Exception:
            logger.exception("Failed to dispatch payload to listener")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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
        path = os.path.join(EMOJI_DIR, p)
        if os.path.exists(path):
            emojis[k] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        else:
            emojis[k] = None
    emojis["no_face"] = np.ones((240, 320, 3), dtype=np.uint8) * 220
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
    """
    Convert an OpenCV image (BGR, BGRA or grayscale) to a QImage.
    Returns a QImage or None on failure.
    """
    if cv_img is None:
        return None
    try:
        h, w = cv_img.shape[:2]
    except Exception:
        return None

    # Grayscale image
    if cv_img.ndim == 2 or (cv_img.ndim == 3 and cv_img.shape[2] == 1):
        fmt = QtGui.QImage.Format_Grayscale8
        qimg = QtGui.QImage(cv_img.data, w, h, cv_img.strides[0], fmt)
        return qimg.copy()

    # BGR -> RGB
    if cv_img.ndim == 3 and cv_img.shape[2] == 3:
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888)
        return qimg.copy()

    # BGRA -> RGBA
    if cv_img.ndim == 3 and cv_img.shape[2] == 4:
        rgba = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
        qimg = QtGui.QImage(rgba.data, w, h, rgba.strides[0], QtGui.QImage.Format_RGBA8888)
        return qimg.copy()

    return None

# Robust representation parser for DeepFace.represent
def _parse_representation(rep):
    try:
        if rep is None:
            return None
        if isinstance(rep, np.ndarray):
            return rep.astype(np.float32)
        if isinstance(rep, (list, tuple)):
            if len(rep) == 0:
                return None
            if all(isinstance(x, (int, float, np.floating, np.integer)) for x in rep):
                return np.asarray(rep, dtype=np.float32)
            return _parse_representation(rep[0])
        if isinstance(rep, dict):
            for key in ("embedding", "represent", "representation", "rep", "vector", "embeddings", "instance"):
                if key in rep:
                    return _parse_representation(rep[key])
            for v in rep.values():
                parsed = _parse_representation(v)
                if parsed is not None:
                    return parsed
        if isinstance(rep, (int, float, np.floating, np.integer)):
            return np.array([rep], dtype=np.float32)
    except Exception:
        logger.exception("Failed to parse representation")
    return None

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
                vec = _parse_representation(rep)
                if vec is None:
                    logger.debug("Could not parse embedding for %s (type=%s)", fpath, type(rep))
                    continue
                vec = l2_normalize(vec)
                emb_list.append(vec)
            except Exception:
                logger.exception("Failed embedding for %s", fpath)
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

# -----------------------------------------------------------------------------
# HTTP Server (SSE + status + dashboard)
# -----------------------------------------------------------------------------
class HttpServer(threading.Thread):
    """
    Flask-based HTTP server that provides:
      - GET /        -> dashboard HTML page (charts)
      - GET /status  -> returns last payload as JSON
      - GET /events  -> Server-Sent Events stream of payloads
    """
    def __init__(self, host="0.0.0.0", port=5000, db=None):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self._app = None
        self._clients = set()
        self._clients_lock = threading.Lock()
        self._last_payload = {}
        self.running = True
        self._db = db
        self._build_app()
        self.start()

    def _build_app(self):
        if Flask is None:
            self._app = None
            logger.warning("Flask not available; HTTP server disabled.")
            return
        app = Flask("moodmirror_http")
        self._app = app

        @app.route("/", methods=["GET"])
        def index():
            try:
                emotions_list = list(EMOTION_IMAGES.keys()) + ["no_face"]
                html = (
                    "<!doctype html><html><head><meta charset='utf-8'>"
                    "<title>MoodMirror Dashboard</title>"
                    "<style>body{font-family:Arial,Helvetica,sans-serif;margin:12px;}canvas{width:100%;height:320px;}</style>"
                    "<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>"
                    "</head><body>"
                    "<h2>MoodMirror - Live Dashboard</h2>"
                    "<p>Emotion distribution (counts) and rolling confidence over time.</p>"
                    "<div style='display:flex;flex-wrap:wrap;gap:12px;'>"
                    "<div style='flex:1 1 480px;min-width:320px;'><canvas id='barChart'></canvas></div>"
                    "<div style='flex:1 1 480px;min-width:320px;'><canvas id='lineChart'></canvas></div>"
                    "</div>"
                    "<script>"
                    "const EMOTIONS = " + json.dumps(emotions_list) + ";"
                    "let counts = {};"
                    "EMOTIONS.forEach(e=>counts[e]=0);"
                    "const barCtx = document.getElementById('barChart').getContext('2d');"
                    "const barChart = new Chart(barCtx,{type:'bar',data:{labels:EMOTIONS,datasets:[{label:'Count',data:EMOTIONS.map(e=>counts[e]),backgroundColor:EMOTIONS.map(()=> 'rgba(54,162,235,0.6)')} ],},options:{responsive:true,maintainAspectRatio:false}});"
                    "const lineCtx = document.getElementById('lineChart').getContext('2d');"
                    "let timeLabels = [], confData = [];"
                    "const lineChart = new Chart(lineCtx,{type:'line',data:{labels:timeLabels,datasets:[{label:'Confidence',data:confData,fill:false,borderColor:'rgba(255,99,132,1)'}]},options:{responsive:true,maintainAspectRatio:false,scales:{y:{min:0,max:100}}}});"
                    "function seedFromStatus(){ fetch('/status').then(r=>r.json()).then(s=>{ try{ if(s && s.emotion){ const emo = s.emotion; counts[emo] = (counts[emo]||0)+1; barChart.data.datasets[0].data = EMOTIONS.map(e=>counts[e]||0); barChart.update(); const label = new Date((s.timestamp||Date.now()/1000)*1000).toLocaleTimeString(); timeLabels.push(label); confData.push(Math.round((s.confidence||0)*100)/100); if(timeLabels.length>30){timeLabels.shift();confData.shift();} lineChart.update(); } }catch(e){console.log('seed err',e);} }).catch(e=>console.log('status fetch err', e)); }"
                    "function connectSSE(){ const es = new EventSource('/events'); es.onopen = function(){ console.log('SSE connected'); }; es.onmessage = function(evt){ try{ const p = JSON.parse(evt.data); const emo = p.emotion || 'neutral'; if(!(emo in counts)) counts[emo]=0; counts[emo] += 1; barChart.data.datasets[0].data = EMOTIONS.map(e=>counts[e]||0); barChart.update(); const label = new Date((p.timestamp||Date.now()/1000)*1000).toLocaleTimeString(); timeLabels.push(label); confData.push(Math.round((p.confidence||0)*100)/100); if(timeLabels.length>30){timeLabels.shift();confData.shift();} lineChart.update(); }catch(err){ console.log('SSE parse err', err); } }; es.onerror = function(e){ console.log('SSE error, reconnecting...'); es.close(); setTimeout(connectSSE,1500); }; }"
                    "seedFromStatus(); connectSSE();"
                    "</script></body></html>"
                )
                return Response(html, mimetype="text/html")
            except Exception:
                return Response("MoodMirror HTTP server\nEndpoints: /status, /events\n", mimetype="text/plain")

        @app.route("/status", methods=["GET"])
        def status():
            try:
                # Option A fallback: prefer db.get_recent_face_moods if present, otherwise do a quick DB read under lock
                try:
                    if self._db is not None and hasattr(self._db, "get_recent_face_moods"):
                        rows = self._db.get_recent_face_moods(minutes=5, limit=1)
                        if rows:
                            return jsonify(self._last_payload or rows[0])
                except Exception:
                    logger.debug("db.get_recent_face_moods check failed", exc_info=True)

                # Fallback to last in-memory payload
                return jsonify(self._last_payload)
            except Exception:
                return jsonify({})

        @app.route("/events", methods=["GET"])
        def events():
            q = queue.Queue()
            with self._clients_lock:
                self._clients.add(q)

            def gen():
                try:
                    yield ": connected\n\n"
                    while True:
                        try:
                            item = q.get(timeout=15.0)
                        except queue.Empty:
                            yield ": heartbeat\n\n"
                            continue
                        if item is None:
                            break
                        try:
                            data = json.dumps(item)
                        except Exception:
                            data = json.dumps({str(k): str(v) for k, v in item.items()})
                        yield "data: %s\n\n" % (data,)
                finally:
                    with self._clients_lock:
                        try:
                            self._clients.remove(q)
                        except Exception:
                            pass

            headers = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
            return Response(stream_with_context(gen()), headers=headers)

    def run(self):
        if self._app is None:
            return
        try:
            # use Flask dev server in background thread
            self._app.run(host=self.host, port=self.port, threaded=True, debug=False, use_reloader=False)
        except Exception:
            logger.exception("HTTP server stopped")

    def broadcast(self, payload):
        try:
            self._last_payload = payload
        except Exception:
            pass
        with self._clients_lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(payload)
            except Exception:
                try:
                    q.put(payload, timeout=0.05)
                except Exception:
                    pass

    def stop(self):
        self.running = False
        with self._clients_lock:
            for q in list(self._clients):
                try:
                    q.put_nowait(None)
                except Exception:
                    pass
            self._clients.clear()

# -----------------------------------------------------------------------------
# DB Logger (threaded)
# -----------------------------------------------------------------------------
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
            except Exception:
                logger.exception("DB logging failed")

    def stop(self):
        self.running = False

# -----------------------------------------------------------------------------
# Text-to-Speech (TTS)
# -----------------------------------------------------------------------------
class TextToSpeech(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.running = True
        self.engine = None
        self._init_engine()
        self.start()

    def _init_engine(self):
        if pyttsx3 is None:
            logger.warning("pyttsx3 not available; voice feedback disabled.")
            self.engine = None
            return
        try:
            self.engine = pyttsx3.init()
            try:
                self.engine.setProperty("rate", 160)
                self.engine.setProperty("volume", 1.0)
            except Exception:
                pass
        except Exception:
            logger.exception("Failed to init pyttsx3")
            self.engine = None

    def speak(self, text):
        if text is None:
            return
        try:
            ascii_text = text.encode("ascii", "ignore").decode("ascii")
        except Exception:
            ascii_text = text
        self.q.put(ascii_text)

    def run(self):
        while self.running:
            try:
                text = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if not text:
                continue
            try:
                if self.engine:
                    self.engine.say(text)
                    self.engine.runAndWait()
                else:
                    logger.info("TTS fallback: %s", text)
            except Exception:
                logger.exception("TTS error")

    def stop(self):
        self.running = False
        try:
            while not self.q.empty():
                time.sleep(0.05)
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Camera worker (Haar pre-check + smoothing)
# -----------------------------------------------------------------------------
class CameraWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object)
    emotion_detected = QtCore.pyqtSignal(str, float, str)

    def __init__(self, src=0, known_embeddings=None, db_logger=None, parent=None,
                 face_miss_threshold=6, cascade_scale=1.2, cascade_min_neighbors=4, min_face_size=(30,30)):
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

        self._no_face_counter = 0
        self._no_face_threshold = face_miss_threshold
        self._cascade_scale = cascade_scale
        self._cascade_min_neighbors = cascade_min_neighbors
        self._min_face_size = min_face_size

        try:
            self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if self._face_cascade.empty():
                self._face_cascade = None
        except Exception:
            self._face_cascade = None

    def _detect_face_fast(self, frame):
        if self._face_cascade is None:
            return True
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            target_w = 320
            if w > target_w:
                scale = target_w / float(w)
                small = cv2.resize(gray, (int(w * scale), int(h * scale)))
            else:
                small = gray
            faces = self._face_cascade.detectMultiScale(small, scaleFactor=self._cascade_scale, minNeighbors=self._cascade_min_neighbors, minSize=self._min_face_size)
            return len(faces) > 0
        except Exception:
            return True

    def run(self):
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            logger.error("Failed to open camera %s", self.src)
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
                        small_for_detect = cv2.resize(frame, SMALL_FRAME_SIZE)
                    except Exception:
                        small_for_detect = frame.copy()

                    face_likely = self._detect_face_fast(small_for_detect)

                    if not face_likely:
                        self._no_face_counter += 1
                        if self._no_face_counter >= self._no_face_threshold:
                            self.emotion_history.clear()
                            self.last_emotion = "Detecting..."
                            try:
                                if self.db_logger:
                                    self.db_logger.enqueue_face_mood("Unknown", "no_face", 0.0, 0)
                            except Exception:
                                pass
                            self.emotion_detected.emit("no_face", 0.0, "Unknown")
                        time.sleep(0.001)
                        continue
                    else:
                        self._no_face_counter = 0

                        try:
                            small = cv2.resize(frame, ANALYSIS_SIZE)
                        except Exception:
                            small = frame.copy()

                        try:
                            result = DeepFace.analyze(small, actions=["emotion"], enforce_detection=False, detector_backend=self.detector_backend)
                            if isinstance(result, list):
                                result = result[0]
                            emotions = result.get("emotion", {}) or {}
                            dominant = result.get("dominant_emotion", None)
                            confidence = float(max(emotions.values())) if emotions else 0.0
                        except Exception as e:
                            logger.debug("DeepFace.analyze error: %s", e)
                            time.sleep(0.001)
                            continue

                        if not dominant or confidence <= 0.0:
                            time.sleep(0.001)
                            continue

                        if dominant and confidence >= MIN_CONFIDENCE:
                            self.emotion_history.append(dominant)
                            try:
                                stable = statistics.mode(self.emotion_history)
                            except statistics.StatisticsError:
                                stable = dominant

                            person_name = "Unknown"
                            try:
                                rep = DeepFace.represent(small, model_name=self.model, enforce_detection=False, detector_backend=self.detector_backend)
                                embedding = _parse_representation(rep)
                                if embedding is not None:
                                    p = recognize_embedding(embedding, self.known_embeddings)
                                    if p:
                                        person_name = p
                            except Exception:
                                logger.debug("Representation error", exc_info=True)

                            if stable != self.last_emotion:
                                self.last_emotion = stable
                                if self.db_logger:
                                    try:
                                        self.db_logger.enqueue_face_mood(person_name, stable, confidence, 0)
                                    except Exception:
                                        pass
                            self.emotion_detected.emit(stable, confidence, person_name)
                        else:
                            # low confidence: ignore
                            pass

                time.sleep(0.001)
        finally:
            try:
                cap.release()
            except Exception:
                pass

    def stop(self):
        self.running = False
        self.wait(2000)

# -----------------------------------------------------------------------------
# Main Window
# -----------------------------------------------------------------------------
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MoodMirror - GUI")
        self.resize(980, 700)

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
        self.dashboard_btn = QtWidgets.QPushButton("Open Dashboard")
        self.quit_btn = QtWidgets.QPushButton("Quit")

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.dashboard_btn)
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
            logger.exception("DB connect failed")
        self.db_logger = DBLogger(self.db)
        self.db_logger.start()

        # Known embeddings
        self.known_embeddings = load_known_embeddings("faces", model_name=KNOWN_MODEL, detector_backend=DETECTOR_BACKEND)

        # TTS
        self.tts = TextToSpeech() if ENABLE_TTS else None
        # last spoken times: {(person, emotion): timestamp}
        self._last_spoken = {}

        # HTTP server (for external dashboard)
        self.http_server = None
        if ENABLE_HTTP and Flask is not None:
            try:
                self.http_server = HttpServer(host=HTTP_HOST, port=HTTP_PORT, db=self.db)
            except Exception:
                logger.exception("Failed to start HTTP server")
                self.http_server = None
        elif ENABLE_HTTP:
            logger.warning("Flask not installed; HTTP disabled.")

        self.worker = None

        self.start_btn.clicked.connect(self.start_worker)
        self.stop_btn.clicked.connect(self.stop_worker)
        self.dashboard_btn.clicked.connect(self.open_dashboard)
        self.quit_btn.clicked.connect(self.close)

        self.current_person = "Unknown"
        self.current_emotion = "Detecting..."
        self.current_confidence = 0.0

        self.blank_display = np.ones((240, 320, 3), dtype=np.uint8) * 255

    def open_dashboard(self):
        url = "http://localhost:%d/" % (HTTP_PORT,)
        if self.http_server is None:
            logger.warning("HTTP server not running; attempting to start it now.")
            if Flask is not None:
                try:
                    self.http_server = HttpServer(host=HTTP_HOST, port=HTTP_PORT, db=self.db)
                    time.sleep(0.25)
                except Exception:
                    logger.exception("Failed to start HTTP server on demand")
            else:
                logger.warning("Flask not installed; cannot start dashboard server.")
        try:
            webbrowser.open(url)
        except Exception:
            logger.exception("Failed to open browser")

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

        # Handle no_face separately in the display
        if emotion == "no_face":
            self.info_label.setText("No face detected. Please look at the camera.")
            emoji_cv = EMOJIS.get("no_face", self.blank_display)
        else:
            self.info_label.setText("Person: %s\nEmotion: %s (%.0f%%)" % (self.current_person, self.current_emotion, self.current_confidence))
            emoji_cv = EMOJIS.get(self.current_emotion, None)

        if emoji_cv is not None:
            try:
                if emotion == "no_face":
                    img = emoji_cv
                else:
                    img = cv2.cvtColor(emoji_cv, cv2.COLOR_BGRA2RGBA) if emoji_cv.shape[2] == 4 else cv2.cvtColor(emoji_cv, cv2.COLOR_BGR2RGB)
                q = cvimg_to_qimage(img)
                pix = QtGui.QPixmap.fromImage(q).scaled(self.emoji_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                self.emoji_label.setPixmap(pix)
            except Exception:
                self.emoji_label.clear()
        else:
            self.emoji_label.clear()

        payload = {
            "person": self.current_person,
            "emotion": self.current_emotion,
            "confidence": float(self.current_confidence),
            "timestamp": time.time()
        }

        if emotion == "no_face":
            payload["emotion"] = "no_face"
            payload["confidence"] = 0.0

        # Notify in-process listeners
        try:
            _notify_listeners(payload)
        except Exception:
            logger.exception("Failed to notify in-process listeners")

        # Broadcast to HTTP clients
        if self.http_server:
            try:
                self.http_server.broadcast(payload)
            except Exception:
                logger.exception("Failed to broadcast payload to HTTP clients")

        # Voice feedback: skip when no_face
        if ENABLE_TTS and self.tts and emotion != "no_face":
            now = time.time()
            key = (self.current_person, self.current_emotion)
            last = self._last_spoken.get(key, 0)
            if now - last >= TTS_SPEAK_INTERVAL:
                templates = PHRASES.get(self.current_emotion, [])
                if not templates:
                    templates = ["You look %s today." % (self.current_emotion,)]
                phrase = random.choice(templates)
                if self.current_person and self.current_person != "Unknown":
                    try:
                        person_ascii = self.current_person.encode("ascii", "ignore").decode("ascii")
                        if "%s" in phrase:
                            try:
                                phrase = phrase % person_ascii
                            except Exception:
                                phrase = "%s, %s" % (person_ascii, phrase)
                        else:
                            phrase = "%s, %s" % (person_ascii, phrase)
                    except Exception:
                        pass
                try:
                    phrase = phrase.encode("ascii", "ignore").decode("ascii")
                except Exception:
                    pass
                try:
                    self.tts.speak(phrase)
                    self._last_spoken[key] = now
                except Exception:
                    logger.exception("Failed to enqueue TTS")

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
            if self.tts:
                self.tts.stop()
        except Exception:
            pass
        try:
            if self.http_server:
                self.http_server.stop()
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

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()