import argparse
import cv2
import imutils
from math import atan2, degrees
from ImageProcessor.transform import four_point_transform
import numpy as np
from skimage.filters import threshold_local

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Input image path")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
ratio = img.shape[0] / 500.0
orig = img.copy()
img = imutils.resize(img, width=500)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# cv2.imshow("Image", img)
# cv2.imshow("Edges", edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

cnt = None
for c in cnts:
    peri = cv2.arcLength(c, closed=True)
    approx = cv2.approxPolyDP(c, 0.04*peri, closed=True)

    if len(approx) == 4:
        cnt = approx
        break

top_left, top_right, bottom_right, bottom_left = cnt.reshape(4, 2)
processed = cv2.drawContours(img.copy(), [cnt], -1, (0, 255, 0), thickness=1)
cv2.circle(processed, tuple(top_left), 7, (255, 0, 0), thickness=-1)
cv2.circle(processed, tuple(bottom_right), 7, (255, 0, 0), thickness=-1)
cv2.circle(processed, tuple(bottom_left), 7, (255, 0, 0), thickness=-1)
cv2.circle(processed, tuple(top_right), 7, (255, 0, 0), thickness=-1)

warped = four_point_transform(img, cnt.reshape(4,2)*ratio)

cv2.imshow('Contoured', imutils.resize(processed, height=650))
cv2.imshow('Transformed', imutils.resize(warped, height=650))
cv2.waitKey(0)
cv2.destroyAllWindows()

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 8, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

# show the original and scanned images
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
cv2.destroyAllWindows()