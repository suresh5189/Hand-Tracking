# -------Initializing the hand’s landmarks detection model using Mediapipe:------------
# Whenever we talk about the detection whether it is an object, person, animal, or as in our case hands detection then the very first step is to initialize the model with valid parameters no matter what detection technique we are following it can either be Mediapipe or Yolo but initializing the model is important, following the same principle we will be following all the given steps:

# # First step is to initialize the Hands class an store it in a variable
# mp_hands = mp.solutions.hand

# # Now second step is to set the hands function which will hold the landmarks points
# hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

# # Last step is to set up the drawing function of hands landmarks on the image
# mp_drawing = mp.solutions.drawing_utils
# Code-breakdown:

# Firstly initializing the class of hands by mp.solutions.hands with a variable.
# Then using the same variable setting up the function for hands by mp.solutions.hands.Hands().

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils 


previousTime = 0
curentTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # print(imgRGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks) 
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lms in enumerate(handLms.landmark):
                # print(id,lms)
                h,w,c = img.shape
                cx,cy = int(lms.x*w),int(lms.y*h)
                print(id,cx,cy)
                cv2.circle(img,(cx,cy),7,(255,0,255),cv2.FILLED)
                
                
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
            
    curentTime = time.time()
    FPS = 1/(curentTime-previousTime)
    previousTime = curentTime
    
    cv2.putText(img,str(int(FPS)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
            
    
    cv2.imshow("Output", img)
    cv2.waitKey(1)