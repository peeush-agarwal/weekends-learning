import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Image path')
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

thresh = cv2.threshold(blurred,60, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow('Thresh',thresh)
# cv2.waitKey(0)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
    M = cv2.moments(c)
    
    # Compute the center 
    cX = int(M["m10"]/M["m00"])
    cY = int(M["m01"]/M["m00"])
    
    cv2.drawContours(img, [c], -1, (0,255,0), thickness=2)
    cv2.circle(img, (cX, cY), 7, (255, 255, 255), thickness=-1)
    cv2.putText(img, "center", (cX-20, cY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)
    cv2.imshow('Contours', img)
    cv2.waitKey(0)
