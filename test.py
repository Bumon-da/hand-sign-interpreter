import cv2
import mediapipe as mp

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.30
)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def distance(a, b):
    return ((a.x - b.x)**2 + (a.y - b.y)**2)**0.5

def classify_number(landmarks):
    tips_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    fingers_up = []

    # Check finger status (excluding thumb)
    for i in range(1, 5):  # Index to Pinky
        tip_y = landmarks[tips_ids[i]].y
        pip_y = landmarks[tips_ids[i] - 2].y
        fingers_up.append(tip_y < pip_y)

    # Thumb logic
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_straight = abs(thumb_tip.x - thumb_ip.x) > 0.04

    # Thumb touching fingers (for 6–9)
    dist_thumb_pinky = distance(landmarks[4], landmarks[20])
    dist_thumb_ring = distance(landmarks[4], landmarks[16])
    dist_thumb_middle = distance(landmarks[4], landmarks[12])
    dist_thumb_index = distance(landmarks[4], landmarks[8])

    # Finger touch threshold (tuned experimentally)
    touch_thresh = 0.05

    # Rules
    if fingers_up == [False, False, False, False] and not thumb_straight:
        return "0"
    elif fingers_up == [True, False, False, False] and not thumb_straight:
        return "1"
    elif fingers_up == [True, True, False, False] and not thumb_straight:
        return "2"
    elif fingers_up == [True, True, False, False] and thumb_straight:
        return "3"
    elif fingers_up == [True, True, True, True] and not thumb_straight:
        return "4"
    elif fingers_up == [True, True, True, True] and thumb_straight:
        return "5"
    elif dist_thumb_pinky < touch_thresh:
        return "6"
    elif dist_thumb_ring < touch_thresh:
        return "7"
    elif dist_thumb_middle < touch_thresh:
        return "8"
    elif dist_thumb_index < touch_thresh:
        return "9"
    elif fingers_up == [False, False, False, False] and thumb_straight:
        return "10"  # Like a thumbs up

    return None

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            number = classify_number(handLms.landmark)
            if number:
                cv2.putText(img, f"Detected Number: {number}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    cv2.imshow("ASL Number Interpreter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
