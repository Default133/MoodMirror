import cv2
import os
import time
import numpy as np
from collections import deque
import statistics
from deepface import DeepFace
from database import Database

# Auto emoji loader
def load_and_resize_emoji(path, target_size=(640, 480)):
    """Load and resize an emoji image while preserving transparency."""
    emoji = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if emoji is None:
        print(f" Warning: Failed to load '{path}'.")
        return None
    return cv2.resize(emoji, target_size, interpolation=cv2.INTER_AREA)

def load_emojis_from_folder(folder_path=".", target_size=(640, 480)):
    """Automatically load all PNGs and map them to emotions by filename."""
    emotion_keywords = {
        "happy": ["smile", "happy", "joy", "laugh"],
        "sad": ["sad", "cry"],
        "surprise": ["shock", "surprise", "wow"],
        "angry": ["angry", "mad", "rage"],
        "neutral": ["neutral", "normal"],
        "devious": ["devious", "smirk", "evil"],
    }

    emojis = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png"):
            filepath = os.path.join(folder_path, filename)
            matched_emotion = None

            for emotion, keywords in emotion_keywords.items():
                if any(keyword in filename.lower() for keyword in keywords):
                    matched_emotion = emotion
                    break

            if matched_emotion:
                emojis[matched_emotion] = load_and_resize_emoji(filepath, target_size)
                print(f" Loaded '{filename}' as '{matched_emotion}' emoji.")
            else:
                print(f" Skipped '{filename}'- no matching emotion keyword found.")
    return emojis

# Overlay function (optimized)
def overlay_emoji(base_img, emoji):
    if emoji is None:
        return base_img
    bg = base_img.copy()

    if emoji.shape[2] == 4:  # Alpha channel
        alpha = emoji[:, :, 3] / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=-1)
        bg = (1 - alpha) * bg + alpha * emoji[:, :, :3]
        return bg.astype(np.uint8)
    else:
        return emoji

# Main Function
def main():
    db = Database(
        host="localhost",
        port=1025,
        user="root",
        password="root",
        database="emoji_tracker"
    )
    db.connect()

    # Load emojis automatically
    emojis = load_emojis_from_folder(".")  # or specify folder like "emojis/"
    if not emojis:
        print(" No emojis loaded! Please check your folder.")
        return

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Emoji Display", cv2.WINDOW_NORMAL)

    emotion_history = deque(maxlen=5)
    last_emotion = None
    display_until = 0
    blank_display = np.full((480, 640, 3), 255, dtype=np.uint8)

    frame_skip = 3
    frame_count = 0
    last_log_time = 0
    log_interval = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)

        if frame_count % frame_skip != 0:
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        try:
            small_frame = cv2.resize(frame, (320, 240))
            result = DeepFace.analyze(
                small_frame, actions=['emotion'], enforce_detection=False
            )

            detected_emotion = result[0]['dominant_emotion']
            confidence = float(max(result[0]['emotion'].values()))

            if confidence < 80:
                continue

            emotion_history.append(detected_emotion)
            stable_emotion = statistics.mode(emotion_history)
            current_time = time.time()

            if stable_emotion != last_emotion:
                print(f"Detected emotion: {stable_emotion}")
                last_emotion = stable_emotion
                display_until = current_time + 2.0

            if current_time - last_log_time >= log_interval:
                duration = current_time - last_log_time
                db.log_mood(stable_emotion, confidence, duration)
                print(f" Logged {stable_emotion} ({confidence:.1f}%) for {duration:.1f}s")
                last_log_time = current_time

        except Exception as e:
            print("Emotion detection error:", e)
            continue

        # Display the emoji
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
