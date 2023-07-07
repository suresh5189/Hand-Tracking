import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

previousTime = 0
curentTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetection()

while True:
    success, img = cap.read()
    img = detector.findhands(img)
    lmList = detector.findPosition(img)
    if len(lmList) !=0:
        print(lmList[0])
        
    curentTime = time.time()
    FPS = 1/(curentTime-previousTime)
    previousTime = curentTime
    
    cv2.putText(img,str(int(FPS)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    
    cv2.imshow("Output", img)
    cv2.waitKey(1)