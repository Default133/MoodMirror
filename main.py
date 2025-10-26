import cv2
import time
import numpy as np
from collections import deque
from deepface import DeepFace
from database import Database

# Load emoji images (ensure they are .png with transparent backgrounds)
emojis = {
    "happy": cv2.imread("smile.png", cv2.IMREAD_UNCHANGED),
    "surprise": cv2.imread("shocked.png", cv2.IMREAD_UNCHANGED),
    "devious": cv2.imread("devious.png", cv2.IMREAD_UNCHANGED),
}

# --- Overlay function for transparent PNGs ---
def overlay_emoji(base_img, emoji):
    if emoji is None:
        return base_img
    emoji_resized = cv2.resize(emoji, (640, 480))
    eh, ew, ec = emoji_resized.shape
    bg = base_img.copy()
    if ec == 4:  # alpha channel
        alpha = emoji_resized[:, :, 3] / 255.0
        for c in range(3):
            bg[:, :, c] = bg[:, :, c] * (1 - alpha) + emoji_resized[:, :, c] * alpha
    else:
        bg = emoji_resized
    return bg

# --- Main Function ---
def main():
    # Initialize database
    db = Database(host="localhost", port=1025, user="root", password="root", database="emoji_tracker")
    db.connect()

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Emoji Display", cv2.WINDOW_NORMAL)

    blank_display = 255 * np.ones((480, 640, 3), dtype=np.uint8)
    emotion_history = deque(maxlen=5)
    last_emotion = None
    display_until = 0

    frame_skip = 3
    frame_count = 0

    last_log_time = time.time()
    log_interval = 1.0  # seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame = cv2.flip(frame, 1)

        if frame_count % frame_skip != 0:
            cv2.imshow("Camera", frame)
            cv2.imshow("Emoji Display", blank_display)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        try:
            # Analyze frame
            small_frame = cv2.resize(frame, (320, 240))
            result = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False)

            detected_emotion = result[0]['dominant_emotion']
            confidence = float(max(result[0]['emotion'].values()))

            # Filter low-confidence predictions
            if confidence < 70:
                continue

            # Append to history and get stable emotion
            emotion_history.append(detected_emotion)
            stable_emotion = max(set(emotion_history), key=emotion_history.count)
            current_time = time.time()

            # Log only when emotion changes or at intervals
            if stable_emotion != last_emotion or current_time - last_log_time >= log_interval:
                duration = current_time - last_log_time if last_emotion == stable_emotion else 0
                db.log_mood(mood=stable_emotion, confidence=confidence, duration=duration)
                last_emotion = stable_emotion
                last_log_time = current_time
                display_until = current_time + 2.0  # show emoji for 2 seconds

        except Exception as e:
            print("Emotion detection error:", e)
            continue

        # Display emoji
        if time.time() < display_until:
            emoji_img = emojis.get(last_emotion, None)
            emoji_display = overlay_emoji(blank_display, emoji_img)
        else:
            emoji_display = blank_display.copy()

        cv2.imshow("Camera", frame)
        cv2.imshow("Emoji Display", emoji_display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    db.close()


if __name__ == "__main__":
    main()
