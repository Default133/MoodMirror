import cv2
import time
import numpy as np
from collections import deque
import statistics
from deepface import DeepFace
from database import Database 

# Load emoji images (ensure they are .png with transparent backgrounds)
emojis = {
    "happy": cv2.imread("smile.png", cv2.IMREAD_UNCHANGED),
    #"sad": cv2.imread("sad.png", cv2.IMREAD_UNCHANGED),
    "surprise": cv2.imread("shocked.png", cv2.IMREAD_UNCHANGED),
    #"angry": cv2.imread("angry.png", cv2.IMREAD_UNCHANGED),
    #"neutral": cv2.imread("neutral.png", cv2.IMREAD_UNCHANGED),
    "devious": cv2.imread("devious.png", cv2.IMREAD_UNCHANGED),
}

# --- Overlay function for transparent PNGs ---
def overlay_emoji(base_img, emoji):
    if emoji is None:
        return base_img
    emoji_resized = cv2.resize(emoji, (640, 480))
    eh, ew, ec = emoji_resized.shape
    bg = base_img.copy()

    if ec == 4:  # includes alpha channel
        alpha = emoji_resized[:, :, 3] / 255.0
        for c in range(3):
            bg[:, :, c] = bg[:, :, c] * (1 - alpha) + emoji_resized[:, :, c] * alpha
    else:
        bg = emoji_resized
    return bg

def log_mood(self, mood, confidence, duration):
    if not self.conn:
        print(" No active database connection. Skipping log.")
        return
    try:
        self.cursor.execute(
            "INSERT INTO moods (mood, confidence, duration) VALUES (%s, %s, %s)",
            (mood, confidence, duration)
        )
        self.conn.commit()
        print(f" Logged mood: {mood} | Confidence: {confidence:.1f}% | Duration: {duration:.2f}s")
    except Exception as e:
        print(f" Failed to log mood: {e}")


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

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Emoji Display", cv2.WINDOW_NORMAL)

    # Emotion stability system
    emotion_history = deque(maxlen=5)
    last_emotion = None
    display_until = 0
    blank_display = (255 * np.ones((480, 640, 3), dtype=np.uint8))

    frame_skip = 3  # analyze every 3th frame
    frame_count = 0

    last_log_time = 0
    log_interval = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)  # horizontal flip (mirror effect)

        # Process every few frames for stability
        if frame_count % frame_skip != 0:
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        try:
            # Downscale to speed up processing
            small_frame = cv2.resize(frame, (320, 240))
            result = DeepFace.analyze(
                small_frame, actions=['emotion'], enforce_detection=False
            )

            detected_emotion = result[0]['dominant_emotion']
            confidence = max(result[0]['emotion'].values())

            # Filter out low-confidence detections
            if confidence < 80:
                continue

            emotion_history.append(detected_emotion)
            stable_emotion = statistics.mode(emotion_history)
            current_time = time.time()

            if stable_emotion != last_emotion:
                print(f"Detected emotion: {stable_emotion}")
                db.log_mood(stable_emotion)
                last_emotion = stable_emotion
                display_until = current_time + 2.0  # smoother transitions

            if current_time - last_log_time >= log_interval:
            # Calculate how long the emotion has lasted
                if last_emotion == stable_emotion:
                    duration = current_time - last_log_time
                else:
                    duration = 0  # emotion just changed

            # Get the confidence for that emotion
                confidence = result[0]['emotion'][stable_emotion]

                print(f" Logging emotion: {stable_emotion}")
                db.log_mood(stable_emotion, confidence, duration)
                last_log_time = current_time


        except Exception as e:
            print("Emotion detection error:", e)
            continue

        # Display logic
        if time.time() < display_until:
            emoji_img = emojis.get(last_emotion, None)
            emoji_display = overlay_emoji(blank_display, emoji_img)
        else:
            emoji_display = blank_display.copy()

        # Show output windows
        cv2.imshow("Camera", frame)
        cv2.imshow("Emoji Display", emoji_display)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
    db.close()

if __name__ == "__main__":
    main()
