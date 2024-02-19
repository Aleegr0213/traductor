import cv2
import mediapipe as mp
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                landmarks_direction = {}
                for id, landmark in enumerate(hand_landmarks.landmark):
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    landmarks_direction[f"Point_{id}"] = (cx, cy)
            
                landmarks_list = [{"Point": point, "x": xy[0], "y": xy[1]} 
                                  for point, xy in landmarks_direction.items()]
                df = pd.DataFrame(landmarks_list)

                print(df)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF ==27:
            break

cap.release()
cv2.destroyAllWindows()

