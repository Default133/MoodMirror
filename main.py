import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import mysql.connector



db = mysql.connector.connect(

    host="localhost",
    port = 1025,       
    user="root",             # Your MySQL username
    password="root",# <-- Change this!
    database="emoji_tracker" # The database you created
)

cursor = db.cursor()


def log_expression(expression_name):
    """Insert expression data into MySQL"""
    query = "SELECT * FROM expressions_log;"
    cursor.execute(query, (expression_name,))
    db.commit()

# Initialize Mediapipe Face Mesh(Has 468 landmarks that keep track of your facial features)
mp_face_mesh = mp.solutions.face_mesh
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)


face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


# Load emoji images
winking_img = cv2.imread("winking.png", cv2.IMREAD_UNCHANGED) #stores the wink image in a variable without changing its properties e.g png
smile_img = cv2.imread("smile.png", cv2.IMREAD_UNCHANGED)     #stores the smile image in a variable without changing its properties
devious_img = cv2.imread("devious.png", cv2.IMREAD_UNCHANGED)
shocked_img = cv2.imread("shocked.png", cv2.IMREAD_UNCHANGED)

# Overlay function (for transparency support) - Also ensures the image has the right color channels i.e RGBA (A is alpha channel which determines the opacity of a pixel) 
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


# Wink detection
wink_frames = 0
def is_winking(landmarks):
    global wink_frames
    left_eye = [33, 160, 158, 133]
    right_eye = [362, 385, 387, 263]

    def eye_aspect(eye_indices):
        top = (landmarks[eye_indices[1]].y + landmarks[eye_indices[2]].y) / 2
        bottom = landmarks[eye_indices[0]].y
        vertical_dist = abs(top - bottom) #Helps in determining the height of an eyelid which is crucial for the wink detection
        horizontal_dist = abs(landmarks[eye_indices[3]].x - landmarks[eye_indices[0]].x) #Eye width
        return vertical_dist / horizontal_dist

    left_ratio = eye_aspect(left_eye)
    right_ratio = eye_aspect(right_eye)
    ratio_diff = abs(left_ratio - right_ratio)

    # less sensitive wink
    if ratio_diff > 0.08:
        wink_frames += 1 
    else:
        wink_frames = max(0, wink_frames - 1)

    return wink_frames > 3 
    
    #This helps to determine whether a proper wink has occured by capturing the frames that have a probable wink


# Smile detection
def is_smiling(landmarks):
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]

    mouth_width = abs(right_mouth.x - left_mouth.x)
    mouth_height = abs(top_lip.y - bottom_lip.y)
    return (mouth_height / mouth_width) > 0.25

def is_shocked(landmarks):
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    mouth_open = abs(top_lip.y - bottom_lip.y)
    return mouth_open > 0.045

def is_hand_detected(results_hands):
    return results_hands.hand_landmarks is not None and len(results_hands.hand_landmarks) > 0


# Main
def main():
    cap = cv2.VideoCapture(0) # Opens the computer's camera
    
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL) #displays the camera window
    cv2.namedWindow("Emoji Display", cv2.WINDOW_NORMAL) #displays the 'emoji' window

    last_expression = None # Tracks the last expression detected
    display_until = 0   #Tracks the time limit that an expression image can be displayed for
    blank_display = (255 * np.ones((480, 640, 3), dtype=np.uint8)) # White background

    while cap.isOpened():
        ret, frame = cap.read() # Keeps track of frames
        if not ret:
            break

        frame = cv2.flip(frame, 1) #Flips the camera horizontally(mirror image)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results_hands = detector.detect(mp_image)

        if results_hands.hand_landmarks:

            for hand_landmarks in results_hands.hand_landmarks:
                h, w, _ = frame.shape

            for landmark in hand_landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),     # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),     # Index
            (5, 9), (9, 10), (10, 11), (11, 12),# Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (0, 17)                             # Palm base connection
        ]
            for start, end in connections:
                x1, y1 = int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h)
                x2, y2 = int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        current_time = time.time()
        smiling_now = False


        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if is_winking(face_landmarks.landmark):
                    last_expression = "wink"
                    display_until = current_time + 1.5  # hold for 1.5 sec
                
                elif is_smiling(face_landmarks.landmark):
                    last_expression = "smile"
                    smiling_now = True
                    display_until = current_time + 1.5
                    
                elif is_shocked(face_landmarks.landmark):
                    last_expression = "shocked"
                    display_until = current_time + 1.5
            if is_hand_detected(results_hands) and smiling_now:
                last_expression = "devious"
                display_until = current_time + 1.5

        # Determine which image to show
        if current_time < display_until:
            if last_expression == "wink":
                emoji_display = overlay_emoji(blank_display, winking_img)
            elif last_expression == "smile":
                emoji_display = overlay_emoji(blank_display, smile_img)
            elif last_expression == "shocked":
                emoji_display = overlay_emoji(blank_display, shocked_img)
            elif last_expression == "devious":
                emoji_display = overlay_emoji(blank_display, devious_img)
            else:
                emoji_display = blank_display.copy()
        else:
            emoji_display = blank_display.copy()

        cv2.imshow("Camera", frame)
        cv2.imshow("Emoji Display", emoji_display)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
