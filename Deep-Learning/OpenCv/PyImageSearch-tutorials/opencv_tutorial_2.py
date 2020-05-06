import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image')
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', img)
cv2.imshow('Gray', gray)

# Edge detection
# applying edge detection we can find the outlines of objects in images
edged = cv2.Canny(gray, 30, 150)
cv2.imshow('Edged', edged)

# Thresholding
# threshold the image by setting all pixel values less than 225
# to 255 (white; foreground) and all pixel values >= 225 to 0
# (black; background), thereby segmenting the image
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow('Threshold', thresh)

# Detecting and drawing contours
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = img.copy()

for c in cnts:
    cv2.drawContours(output, [c], -1, (240, 0, 159), thickness=3)
    cv2.imshow('Contours', output)
    cv2.waitKey(0)

text = f'Total objects: {len(cnts)}'
cv2.putText(output, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 0, 159), thickness=2)
cv2.imshow('Contours', output)

# Erosions and dilations
# we apply erosions to reduce the size of foreground objects
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow('Erode', mask)

# Similarly, we can apply dilations to enlarge the size of foreground objects
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow('Dilated', mask)

# Masking and bitwise operations
# a typical operation we may want to apply is to take our mask and
# apply a bitwise AND to our input image, keeping only the masked
# regions
mask = thresh.copy()
output = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('Masked', output)

cv2.waitKey(0)