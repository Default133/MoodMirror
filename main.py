import cv2
import time
import numpy as np
from collections import deque
import statistics
from deepface import DeepFace
from database import Database
import warnings
import logging
import os

warnings.filterwarnings("ignore")
logging.getLogger("deepface").setLevel(logging.ERROR)
logging.basicConfig(filename="deepface_log.txt", level=logging.ERROR)

# Emotion -> Image mapping
emotion_images = {
    "happy": "happy.png",
    "surprise": "surprise.png",
    "disgust": "disgust.png",
    "sad": "sad.png",
    "angry": "angry.png",
    "neutral": "neutral.png",
    "fear": "fear.png"
}

# Load available images safely
emojis = {}
for emotion, path in emotion_images.items():
    if os.path.exists(path):
        emojis[emotion] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        emojis[emotion] = None

def overlay_emoji(base_img, emoji):
    if emoji is None:
        return base_img
    emoji_resized = cv2.resize(emoji, (640, 480))
    eh, ew, ec = emoji_resized.shape
    bg = base_img.copy()
    if ec == 4:
        alpha = emoji_resized[:, :, 3] / 255.0
        for c in range(3):
            bg[:, :, c] = bg[:, :, c] * (1 - alpha) + emoji_resized[:, :, c] * alpha
    else:
        bg = emoji_resized
    return bg

def main():
    db = Database(
        host="localhost",
        port=1025,
        user="root",
        password="root",
        database="emoji_tracker"
    )
    db.connect()

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Emoji Display", cv2.WINDOW_NORMAL)

    emotion_history = deque(maxlen=10)
    display_duration = 3.0
    min_confidence = 75.0
    frame_skip = 10
    small_frame_size = (160, 120)
    blank_display = np.ones((480, 640, 3), dtype=np.uint8) * 255

    last_emotion = "Detecting..."
    last_person = "Unknown"
    display_until = 0
    frame_count = 0
    last_log_time = 0
    log_interval = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)
        current_time = time.time()

        if frame_count % frame_skip == 0:
            try:
                small_frame = cv2.resize(frame, small_frame_size)
                result = DeepFace.analyze(small_frame, actions=["emotion"], enforce_detection=False)

                detected_emotion = result[0]["dominant_emotion"]
                confidence = float(max(result[0]["emotion"].values()))
                if confidence < min_confidence:
                    continue

                emotion_history.append(detected_emotion)
                stable_emotion = statistics.mode(emotion_history)

                try:
                    recognition = DeepFace.find(small_frame, db_path="faces", enforce_detection=False)
                    if len(recognition) > 0 and len(recognition[0]) > 0:
                        person_path = recognition[0]["identity"][0]
                        last_person = person_path.split("\\")[-2]
                except Exception:
                    pass

                if stable_emotion != last_emotion:
                    print(f"Detected {last_person} feeling {stable_emotion} ({confidence:.1f}%)")
                    db.log_mood(stable_emotion, confidence, 0)
                    last_emotion = stable_emotion
                    display_until = current_time + display_duration

                if current_time - last_log_time >= log_interval:
                    duration = current_time - last_log_time if stable_emotion == last_emotion else 0
                    db.log_mood(stable_emotion, confidence, duration)
                    last_log_time = current_time

            except Exception as e:
                print("Emotion detection error:", e)
                continue

        if time.time() < display_until and last_emotion in emojis:
            emoji_img = emojis[last_emotion]
            emoji_display = overlay_emoji(blank_display, emoji_img)
        else:
            emoji_display = blank_display.copy()

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (480, 65), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        label = f"{last_person}: {last_emotion}"
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Camera", frame)
        cv2.imshow("Emoji Display", emoji_display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    db.close()

if __name__ == "__main__":
    main()
