import cv2
import imutils
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Input image path')
args = vars(ap.parse_args())

img = cv2.imread(args["image"])

# Loop over rotation angles
for angle in np.arange(0, 360, 15):
    rotated = imutils.rotate(img, angle)
    cv2.imshow('Rotated', rotated)
    cv2.waitKey(0)
    
# Loop over rotation angles, but this time no corners are cut off
for angle in np.arange(0, 360, 15):
    rotated = imutils.rotate_bound(img, angle)
    cv2.imshow('Rotated', rotated)
    cv2.waitKey(0)