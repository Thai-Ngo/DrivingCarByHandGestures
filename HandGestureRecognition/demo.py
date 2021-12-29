import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

camera = cv2.VideoCapture(0)

hand = mp_hands.Hands(
  min_detection_confidence=0.5, 
  min_tracking_confidence=0.5,
  max_num_hands=1
)
try:
  while camera.isOpened():
    ret, frame = camera.read()

    if not ret:
      print("Can't open camera.")
      break
    # Mark the image as not writeable to pass by reference
    frame.flags.writeable = False 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand.process(frame)
    
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
        frame,
        (results.multi_hand_landmarks)[0],
        mp_hands.HAND_CONNECTIONS
        # mp_drawing_styles.get_default_hand_landmarks_style(),
        # mp_drawing_styles.get_default_hand_connections_style()
      )
    

    cv2.imshow('MediaPipe Hands', cv2.flip(frame, 1))
    
    cv2.waitKey(1)
  
finally:
  camera.release()