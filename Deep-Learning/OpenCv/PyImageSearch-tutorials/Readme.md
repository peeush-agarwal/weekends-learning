# OpenCv in Python

## Getting started

### [Install opencv](https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/)

`pip install opencv-contrib-python`

### [Command line arguments](https://www.pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/)

Command line arguments are flags given to a program/script at runtime. They contain additional information for our program so that it can execute.

#### Why Command line arguments?

+ This allows us to give our program different input on the fly without changing the code.

#### The `argparse` python library

``` {python}
# import the necessary packages
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,
	help="name of the user")
args = vars(ap.parse_args())

# display a friendly message to the user
print("Hi there {}, it's nice to meet you!".format(args["name"]))
```

## Basic operations in OpenCv

### Read an image from disk
`img = cv2.imread(args["input"])`

### Get Height, Width and Depth from cv2 image object
`h, w, d = img.shape`

### View image object
`cv2.imshow(img)`

### Read an individual pixel value from image object
`B, G, R = img[100, 50]`

### Cropping image to fetch **Region of interest**(ROI)
``` {python}
roi = img[60:160, 320:420]
cv2.imshow('ROI', roi)
```

### Resize image
Resizing images is important for number of reasons. 
+ First, you might fit a large image on your screen
+ Fewer pixels to process makes Image processing faster
+ In the case of deep learning, we often resize images, ignoring aspect ratio, so that the volume fits into a network which requires that an image be square and of a certain dimension.

``` {python}
# Method 1: without maintaining aspect ratio from original image
resized = cv2.resize(img, (200, 200))
cv2.imshow('Resized', resized)

# Method 2: calculate aspect ratio from image and then resize
r = 300.0 / w
dim = (300, int(h*r))
resized = cv2.resize(img, dim)
cv2.imshow('Resized', resized)

# Method 3: using imutils package, resize maintains the aspect ratio
resized = imutils.resize(img, width=300)
cv2.imshow('Resized', resized)
```

### Rotate image 45 degree clockwise
``` {python}
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
```

### Smoothing an image
In many image processing pipelines, we must blur an image to reduce high-frequency noise, making it easier for our algorithms to detect and understand the actual contents of the image rather than just noise that will “confuse” our algorithms. 

``` {python}
# Apply a Gaussian blur with a 11x11 kernel to the image to smooth it, 
# useful when reducing high frequency noise
blurred = cv2.GaussianBlur(img, (11,11), 0)
cv2.imshow('Blurred', blurred)
```

### Converting an image to grayscale
`gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`

### Edge detection
Edge detection is useful for finding boundaries of objects in an image — it is effective for segmentation purposes.
``` {python}
# applying edge detection we can find the outlines of objects in
# images
edged = cv2.Canny(gray, 30, 150)
cv2.imshow("Edged", edged)
```

### Thresholding
Thresholding can help us to remove lighter or darker regions and contours of images.
``` {python}
# threshold the image by setting all pixel values less than 225
# to 255 (white; foreground) and all pixel values >= 225 to 255
# (black; background), thereby segmenting the image
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh", thresh)
```

### Detecting and drawing contours
```{python}
# find contours (i.e., outlines) of the foreground objects in the
# thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()
# loop over the contours
for c in cnts:
	# draw each contour on the output image with a 3px thick purple
	# outline, then display the output contours one at a time
	cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
	cv2.imshow("Contours", output)
	cv2.waitKey(0)

# draw the total number of contours found in purple
text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(240, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey(0)
```

### Erosions and dilations
Erosions and dilations are typically used to reduce noise in binary images (a side effect of thresholding).

To reduce the size of foreground objects we can erode away pixels given a number of iterations:
```{python}
# we apply erosions to reduce the size of foreground objects
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow("Eroded", mask)
```

To enlarge the regions, simply use `cv2.dilate`:
```{python}
# similarly, dilations can increase the size of the ground objects
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow("Dilated", mask)
```

### Masking and bitwise operations
Masks allow us to “mask out” regions of an image we are uninterested in. We call them “masks” because they will hide regions of images we do not care about.
```{python}
# a typical operation we may want to apply is to take our mask and
# apply a bitwise AND to our input image, keeping only the masked
# regions
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Output", output)
```