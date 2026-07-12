import cv2
import time
import sys

print("Opening camera...")
cap = cv2.VideoCapture('/dev/video9')
if not cap.isOpened():
    print("Could not open camera!")
    sys.exit(1)

print("Reading frame...")
start = time.time()
ret, frame = cap.read()
end = time.time()

if ret:
    print(f"Successfully read a frame in {end - start:.3f} seconds!")
else:
    print(f"Failed to read frame after {end - start:.3f} seconds!")
