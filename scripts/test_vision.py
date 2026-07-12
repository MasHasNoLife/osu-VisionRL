import cv2
import numpy as np

cap = cv2.VideoCapture('/dev/video9')
ret, frame = cap.read()
if ret:
    cv2.imwrite("vision_raw.png", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    PLAY_LEFT = 320
    PLAY_RIGHT = 2240
    PLAY_TOP = 150
    PLAY_BOTTOM = 1290
    SCREEN_W = 2560
    SCREEN_H = 1440
    
    gray[0:PLAY_TOP, :] = 0              
    gray[PLAY_BOTTOM:SCREEN_H, :] = 0  
    gray[:, 0:PLAY_LEFT] = 0             
    gray[:, PLAY_RIGHT:SCREEN_W] = 0   
    
    cv2.imwrite("vision_masked.png", gray)
    
    playfield = gray[PLAY_TOP:PLAY_BOTTOM, PLAY_LEFT:PLAY_RIGHT]
    resized = cv2.resize(playfield, (84, 84))
    cv2.imwrite("vision_84x84.png", resized)
    print("Saved frames!")
else:
    print("Failed to read frame")
