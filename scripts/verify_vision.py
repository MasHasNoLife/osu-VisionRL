import cv2
import numpy as np
import time

print("You have 5 seconds to switch to the osu! window...")
for i in range(5, 0, -1):
    print(f"{i}...")
    time.sleep(1)

print("Capturing frame...")
cap = cv2.VideoCapture('/dev/video9', cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Read a few frames to clear any old buffered frames
for _ in range(5):
    cap.read()

ret, frame = cap.read()
if ret:
    cv2.imwrite("vision_raw_test.png", frame)
    
    # Process exactly like the AI does
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    PLAY_LEFT = 320
    PLAY_RIGHT = 2240
    PLAY_TOP = 200
    PLAY_BOTTOM = 1240
    SCREEN_W = 2560
    SCREEN_H = 1440
    
    # Masking
    gray[0:PLAY_TOP, :] = 0              
    gray[PLAY_BOTTOM:SCREEN_H, :] = 0  
    gray[:, 0:PLAY_LEFT] = 0             
    gray[:, PLAY_RIGHT:SCREEN_W] = 0   
    
    cv2.imwrite("vision_masked_test.png", gray)
    
    # Cropping and resizing
    playfield = gray[PLAY_TOP:PLAY_BOTTOM, PLAY_LEFT:PLAY_RIGHT]
    resized = cv2.resize(playfield, (84, 84))
    cv2.imwrite("vision_84x84_test.png", resized)
    
    print("Success! Saved vision_raw_test.png, vision_masked_test.png, and vision_84x84_test.png in your project folder.")
else:
    print("Failed to read frame from /dev/video9")
