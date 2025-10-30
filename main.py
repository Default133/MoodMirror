# -*- coding: ascii -*-
"""
Updated main.py - PyQt5 GUI front-end integrated with DeepFace analysis.

This update fixes issues with the dashboard not rendering charts by:
- Improving the SSE endpoint (correct headers, keep-alive heartbeats to avoid buffering).
- Having the dashboard page fetch /status on load to seed charts.
- Making the dashboard HTML/CSS/JS more robust (responsive canvases, reconnect logic).
- Adding safer JSON serialization and small server-side improvements.

Notes:
- Requires: pip install opencv-python deepface PyQt5 mysql-connector-python pyttsx3 flask
- Run: python main.py
"""
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

logging.getLogger("deepface").setLevel(logging.ERROR)
logging.basicConfig(filename="deepface_log.txt", level=logging.ERROR)

# -----------------------------------------------------------------------------
# HTTP server config
# -----------------------------------------------------------------------------
ENABLE_HTTP = True            # set False to disable the HTTP server
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 5000              # clients connect to http://<host>:5000
# Endpoints:
#  - GET /        -> dashboard HTML page (charts)
#  - GET /status  -> last payload as JSON
#  - GET /events  -> Server-Sent Events stream of payloads

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
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

# Voice config
ENABLE_TTS = True             # toggle voice feedback on/off
TTS_SPEAK_INTERVAL = 5.0      # minimum seconds between speaking the same emotion

# Per-emotion phrase templates (ASCII-only).
PHRASES = {
    "happy": [
        "You look very happy today.",
        "Nice to see you smiling.",
        "What a great smile you have today."
    ],
    "surprise": [
        "You seem surprised. Everything ok?",
        "Wow, you look surprised!",
        "That looked like a surprise moment."
    ],
    "disgust": [
        "You look a bit displeased. Is everything all right?",
        "You do not seem pleased. Do you want to try again?",
        "Hmm, that expression looks unhappy. Are you okay?"
    ],
    "sad": [
        "You look a little sad. I am here if you need me.",
        "I am sorry you seem down today.",
        "It seems like a quiet moment. Take your time."
    ],
    "angry": [
        "You look upset. Remember to breathe.",
        "I sense some anger. Want to take a break?",
        "It seems like something is bothering you."
    ],
    "neutral": [
        "You look calm and neutral.",
        "All seems steady right now.",
        "You seem relaxed at the moment."
    ],
    "fear": [
        "You look concerned. Are you okay?",
        "You seem frightened. Take a deep breath.",
        "I notice some worry in your expression."
    ]
}

# -----------------------------------------------------------------------------
# Integration API for in-process dashboards (kept for compatibility)
# -----------------------------------------------------------------------------
_update_listeners = []
_update_listeners_lock = threading.Lock()


def register_update_callback(cb):
    if not callable(cb):
        raise ValueError("callback must be callable")
    with _update_listeners_lock:
        if cb not in _update_listeners:
            _update_listeners.append(cb)


def unregister_update_callback(cb):
    with _update_listeners_lock:
        try:
            _update_listeners.remove(cb)
        except ValueError:
            pass


def _notify_listeners(payload):
    with _update_listeners_lock:
        listeners = list(_update_listeners)
    for cb in listeners:
        try:
            threading.Thread(target=cb, args=(payload,), daemon=True).start()
        except Exception:
            logging.exception("Failed to dispatch payload to listener")


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
    def __init__(self, host="0.0.0.0", port=5000):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self._app = None
        self._clients = set()
        self._clients_lock = threading.Lock()
        self._last_payload = {}
        self.running = True
        self._build_app()
        self.start()

    def _build_app(self):
        if Flask is None:
            self._app = None
            logging.warning("Flask not available; HTTP server disabled.")
            return
        app = Flask("moodmirror_http")
        self._app = app

        @app.route("/", methods=["GET"])
        def index():
            # Dashboard HTML with Chart.js and SSE client.
            # It fetches /status initially, then subscribes to /events.
            try:
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
                    "const EMOTIONS = " + json.dumps(list(EMOTION_IMAGES.keys())) + ";"
                    "let counts = {};"
                    "EMOTIONS.forEach(e=>counts[e]=0);"
                    "const barCtx = document.getElementById('barChart').getContext('2d');"
                    "const barChart = new Chart(barCtx,{type:'bar',data:{labels:EMOTIONS,datasets:[{label:'Count',data:EMOTIONS.map(e=>counts[e]),backgroundColor:EMOTIONS.map(()=> 'rgba(54,162,235,0.6)')} ],},options:{responsive:true,maintainAspectRatio:false}});"
                    "const lineCtx = document.getElementById('lineChart').getContext('2d');"
                    "let timeLabels = [], confData = [];"
                    "const lineChart = new Chart(lineCtx,{type:'line',data:{labels:timeLabels,datasets:[{label:'Confidence',data:confData,fill:false,borderColor:'rgba(255,99,132,1)'}]},options:{responsive:true,maintainAspectRatio:false,scales:{y:{min:0,max:100}}}});"
                    "function seedFromStatus(){"
                    " fetch('/status').then(r=>r.json()).then(s=>{"
                    "   try{ if(s && s.emotion){ const emo = s.emotion; counts[emo] = (counts[emo]||0)+1; barChart.data.datasets[0].data = EMOTIONS.map(e=>counts[e]||0); barChart.update(); const label = new Date((s.timestamp||Date.now()/1000)*1000).toLocaleTimeString(); timeLabels.push(label); confData.push(Math.round((s.confidence||0)*100)/100); if(timeLabels.length>30){timeLabels.shift();confData.shift();} lineChart.update(); } }catch(e){console.log('seed err',e);} }).catch(e=>console.log('status fetch err', e));"
                    "}"
                    "function connectSSE(){"
                    " const es = new EventSource('/events');"
                    " es.onopen = function(){ console.log('SSE connected'); };"
                    " es.onmessage = function(evt){"
                    "   try{ const p = JSON.parse(evt.data); const emo = p.emotion || 'neutral'; if(!(emo in counts)) counts[emo]=0; counts[emo] += 1; barChart.data.datasets[0].data = EMOTIONS.map(e=>counts[e]||0); barChart.update(); const label = new Date((p.timestamp||Date.now()/1000)*1000).toLocaleTimeString(); timeLabels.push(label); confData.push(Math.round((p.confidence||0)*100)/100); if(timeLabels.length>30){timeLabels.shift();confData.shift();} lineChart.update(); }catch(err){ console.log('SSE parse err', err); }"
                    " };"
                    " es.onerror = function(e){ console.log('SSE error, reconnecting...'); es.close(); setTimeout(connectSSE,1500); };"
                    "}"
                    "seedFromStatus(); connectSSE();"
                    "</script></body></html>"
                )
                return Response(html, mimetype="text/html")
            except Exception:
                return Response("MoodMirror HTTP server\nEndpoints: /status, /events\n", mimetype="text/plain")

        @app.route("/status", methods=["GET"])
        def status():
            try:
                return jsonify(self._last_payload)
            except Exception:
                return jsonify({})

        @app.route("/events", methods=["GET"])
        def events():
            # SSE generator queue for this client
            q = queue.Queue()
            with self._clients_lock:
                self._clients.add(q)

            def gen():
                try:
                    # Immediately send a comment to ensure connection is open
                    yield ": connected\n\n"
                    # send items as they arrive; include periodic heartbeats
                    while True:
                        try:
                            item = q.get(timeout=15.0)
                        except queue.Empty:
                            # heartbeat comment to keep intermediaries from buffering
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
                # hint for proxies to disable buffering
                "X-Accel-Buffering": "no"
            }
            return Response(stream_with_context(gen()), headers=headers)

    def run(self):
        if self._app is None:
            return
        try:
            # Run Flask dev server in background thread. For production use gunicorn.
            self._app.run(host=self.host, port=self.port, threaded=True, debug=False, use_reloader=False)
        except Exception as e:
            logging.error("HTTP server stopped: %s", e)

    def broadcast(self, payload):
        # store last payload for /status
        try:
            self._last_payload = payload
        except Exception:
            pass
        # enqueue to all client queues
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
        # notify all clients to terminate
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
            except Exception as e:
                logging.error("DB logging failed: %s", e)

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
            logging.warning("pyttsx3 not available; voice feedback disabled.")
            self.engine = None
            return
        try:
            self.engine = pyttsx3.init()
            try:
                self.engine.setProperty("rate", 160)
                self.engine.setProperty("volume", 1.0)
            except Exception:
                pass
        except Exception as e:
            logging.warning("Failed to initialize pyttsx3 engine: %s", e)
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
                    logging.info("TTS fallback (no engine): %s", text)
            except Exception as e:
                logging.error("TTS error: %s", e)

    def stop(self):
        self.running = False
        try:
            while not self.q.empty():
                time.sleep(0.05)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Camera & Analysis worker (QThread)
# -----------------------------------------------------------------------------
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
            pass
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
                self.http_server = HttpServer(host=HTTP_HOST, port=HTTP_PORT)
            except Exception as e:
                logging.error("Failed to start HTTP server: %s", e)
                self.http_server = None
        elif ENABLE_HTTP:
            logging.warning("HTTP support requested but Flask is not installed. Install 'flask' to enable it.")

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
        # Open dashboard in default browser. Ensure server is started.
        url = "http://localhost:%d/" % (HTTP_PORT,)
        if self.http_server is None:
            logging.warning("HTTP server not running; attempting to start it now.")
            if Flask is not None:
                try:
                    self.http_server = HttpServer(host=HTTP_HOST, port=HTTP_PORT)
                    # small delay to let server bind
                    time.sleep(0.25)
                except Exception as e:
                    logging.error("Failed to start HTTP server on demand: %s", e)
            else:
                logging.warning("Flask not installed; cannot start dashboard server.")
        try:
            webbrowser.open(url)
        except Exception as e:
            logging.error("Failed to open browser: %s", e)

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

        # Prepare payload for dashboard listeners and external clients
        payload = {
            "person": self.current_person,
            "emotion": self.current_emotion,
            "confidence": float(self.current_confidence),
            "timestamp": time.time()
        }

        # Notify in-process listeners
        try:
            _notify_listeners(payload)
        except Exception:
            logging.exception("Failed to notify in-process listeners")

        # Broadcast to HTTP clients (separate process dashboards)
        if self.http_server:
            try:
                self.http_server.broadcast(payload)
            except Exception:
                logging.exception("Failed to broadcast payload to HTTP clients")

        # Voice feedback: speak a phrase chosen for the detected emotion (throttled)
        if ENABLE_TTS and self.tts:
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
                except Exception as e:
                    logging.error("Failed to enqueue TTS: %s", e)

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