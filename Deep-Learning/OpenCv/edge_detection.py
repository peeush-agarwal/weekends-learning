import cv2
import numpy as np

def detect_edges(filename, minThreshVal, maxThreshVal, display_original = False):
    img = cv2.imread(filename, 0)
    edges = cv2.Canny(img, minThreshVal, maxThreshVal)

    if display_original:
        cv2.imshow('Original', img)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    
    return edges

def detect_edges_real_time():
    cap = cv2.VideoCapture(0)

    while(cap.isOpened()):
        ret, image = cap.read()

        if not ret:
            break

        edges = cv2.Canny(image, 100, 200)
        cv2.imshow('Edges', edges)
        
        # Wait for Esc key to stop 
        k = cv2.waitKey(1) & 0xFF
        if k == 27: 
            break
    
    cap.release()
    cv2.destroyAllWindows()

def detect_edges_real_time_advanced():
    # capture frames from a camera 
    cap = cv2.VideoCapture(0) 
  
    # loop runs if capturing has been initialized 
    while(1): 
        # reads frames from a camera 
        ret, frame = cap.read() 
    
        # Display an original image 
        cv2.imshow('Original',frame) 
    
        # converting BGR to HSV 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
        
        cv2.imshow('HSV', hsv)

        # define range of red color in HSV 
        lower_red = np.array([30,150,50]) 
        upper_red = np.array([255,255,180]) 
        
        # create a red HSV colour boundary and  
        # threshold HSV image 
        mask = cv2.inRange(hsv, lower_red, upper_red) 
    
        # Bitwise-AND mask and original image 
        res = cv2.bitwise_and(frame,frame, mask= mask) 
    
        # finds edges in the input image image and 
        # marks them in the output map edges 
        edges = cv2.Canny(frame,100,200) 
    
        # Display edges in a frame 
        cv2.imshow('Edges',edges) 
    
        # Wait for Esc key to stop 
        k = cv2.waitKey(1) & 0xFF
        if k == 27: 
            break
  
    # Close the window 
    cap.release() 
    
    # De-allocate any associated memory usage 
    cv2.destroyAllWindows() 