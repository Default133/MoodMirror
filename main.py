import cv2
import os
import time
import numpy as np
import statistics
from collections import deque
from deepface import DeepFace
from database import Database

# Load emoji images
emojis = {
    "happy": cv2.imread("smile.png", cv2.IMREAD_UNCHANGED),
    "surprise": cv2.imread("shocked.png", cv2.IMREAD_UNCHANGED),
    "devious": cv2.imread("devious.png", cv2.IMREAD_UNCHANGED),
}

def overlay_emoji(base_img, emoji):
    if emoji is None:
        return base_img
    emoji_resized = cv2.resize(emoji, (640, 480))
    if emoji_resized.shape[2] == 4:
        alpha = emoji_resized[:, :, 3] / 255.0
        for c in range(3):
            base_img[:, :, c] = base_img[:, :, c] * (1 - alpha) + emoji_resized[:, :, c] * alpha
    return base_img

# Load known faces for recognition
def load_known_faces(folder="faces"):
    faces = []
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            name = os.path.splitext(file)[0]
            path = os.path.join(folder, file)
            faces.append((name, path))
    return faces

def recognize_face(frame, faces, threshold=0.45):
    try:
        for name, path in faces:
            result = DeepFace.verify(frame, path, enforce_detection=False, model_name="Facenet512")
            if result["verified"] and result["distance"] < threshold:
                return name
    except Exception:
        pass
    return "Unknown"

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

    known_faces = load_known_faces()

    emotion_history = deque(maxlen=10)
    last_emotion = None
    last_person = None
    display_until = 0
    display_duration = 3.0
    blank_display = np.ones((480, 640, 3), dtype=np.uint8) * 255
    frame_skip = 10
    frame_count = 0
    last_log_time = 0
    log_interval = 5
    min_confidence = 75.0

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
            small_frame = cv2.resize(frame, (160, 120))

            # Detect emotion
            result = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False)
            detected_emotion = result[0]['dominant_emotion']
            confidence = float(max(result[0]['emotion'].values()))
            if confidence < min_confidence:
                continue

            emotion_history.append(detected_emotion)
            stable_emotion = statistics.mode(emotion_history)
            current_time = time.time()

            # Recognize face (only when emotion stable)
            person_name = recognize_face(frame, known_faces) if stable_emotion == last_emotion else last_person

            if stable_emotion != last_emotion or person_name != last_person:
                print(f"Detected {person_name} feeling {stable_emotion} ({confidence:.1f}%)")
                db.log_mood(stable_emotion, confidence, 0)
                last_emotion = stable_emotion
                last_person = person_name
                display_until = current_time + display_duration

            if current_time - last_log_time >= log_interval:
                duration = current_time - last_log_time
                db.log_mood(stable_emotion, confidence, duration)
                last_log_time = current_time

        except Exception as e:
            print("Detection error:", e)
            continue

        if time.time() < display_until:
            emoji_img = emojis.get(last_emotion, None)
            emoji_display = overlay_emoji(blank_display.copy(), emoji_img)
        else:
            emoji_display = blank_display.copy()

        if last_emotion:
            label = f"{last_person}: {last_emotion} ({confidence:.1f}%)"
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Camera", frame)
        cv2.imshow("Emoji Display", emoji_display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    db.close()
    print("Program ended.")

if __name__ == "__main__":
    main()
