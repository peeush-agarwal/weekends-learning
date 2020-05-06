import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', help="Input image path")
args = vars(ap.parse_args())

# Read image from input path
img = cv2.imread(args["input"])

# Height, Width and Depth of an image
# - Height = Number of rows
# - Width = Number of columns
# - Depth = Number of channels
h, w, d = img.shape
print(f'Height:{h}, width:{w}, Depth:{d}')

# Show the read image 
cv2.imshow("Original", img)

# Read a pixel at x=50 and y=100 value from image
B, G, R = img[100, 50]
print(f'R:{R} G:{G} B:{B}')

# Array slicing and cropping
# Extracting regions of interest (ROI)
# Extract a 100x100 pixel square ROI from input image starting 
#  at x=320 y=60 and ending at x=420 y=160
roi = img[60:160, 320:420]
cv2.imshow("ROI", roi)

# Resize original image to 200x200. Note cv2 doesn't maintain aspect ratio of original image
resized = cv2.resize(img, (200, 200))
cv2.imshow('Resized', resized)

# Calculate aspect ratio so that resize doesn't lose aspect ratio
r = 300.0 / w
dim = (300, int(h*r))
resized = cv2.resize(img, dim)
cv2.imshow('Resized', resized)

# imutils can resize maintaining the aspect ratio
resized = imutils.resize(img, width=300)
cv2.imshow('Resized', resized)

# Rotating an image
# let's rotate an image 45 degrees clockwise using OpenCV by first
# computing the image center, then constructing the rotation matrix,
# and then finally applying the affine warp
center = (w//2, h//2)
M = cv2.getRotationMatrix2D(center, angle = -45, scale = 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
cv2.imshow('Rotated:45 degree clockwise', rotated)

# imutils rotate method
rotated = imutils.rotate(img, -45)
cv2.imshow('Rotated:45 degree clockwise', rotated)

# OpenCV doesn't "care" if our rotated image is clipped after rotation
# so we can instead use another imutils convenience function to help
# us out
rotated = imutils.rotate_bound(img, 45)
cv2.imshow('Rotated:45 degree clockwise', rotated)

# Apply a Gaussian blur with a 11x11 kernel to the image to smooth it, 
# useful when reducing high frequency noise
blurred = cv2.GaussianBlur(img, (11,11), 0)
cv2.imshow('Blurred', blurred)


# Wait for key to press to close the window
cv2.waitKey(0)