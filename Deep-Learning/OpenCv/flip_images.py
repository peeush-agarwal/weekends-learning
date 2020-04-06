import cv2
import numpy as np

def flip_webcam_frames_vertically():
    cap = cv2.VideoCapture(0)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        flipped = cv2.flip(frame, 0)

        cv2.imshow('Flipped', flipped)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
