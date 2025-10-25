import cv2
import time
import numpy as np
from deepface import DeepFace
from database import Database  # import your database abstraction


emojis = {
    "happy": cv2.imread("smile.png", cv2.IMREAD_UNCHANGED),
    "surprise": cv2.imread("shocked.png", cv2.IMREAD_UNCHANGED),
    "angry": cv2.imread("devious.png", cv2.IMREAD_UNCHANGED),
}

# Overlay function for emoji transparency
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
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Emoji Display", cv2.WINDOW_NORMAL)

    db = Database()
    db.connect()

    last_emotion = None
    display_until = 0
    blank_display = (255 * np.ones((480, 640, 3), dtype=np.uint8))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror view

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']

            current_time = time.time()

            if dominant_emotion != last_emotion:
                print(f"Detected emotion: {dominant_emotion}")
                db.log_mood(dominant_emotion)
                last_emotion = dominant_emotion
                display_until = current_time + 1.5

        except Exception as e:
            print("Emotion detection error:", e)
            dominant_emotion = None

        # Show emoji if relevant
        current_time = time.time()
        if current_time < display_until and last_emotion in emojis:
            emoji_display = overlay_emoji(blank_display, emojis[last_emotion])
        else:
            emoji_display = blank_display.copy()

        cv2.imshow("Camera", frame)
        cv2.imshow("Emoji Display", emoji_display)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    db.close()


if __name__ == "__main__":
    main()
