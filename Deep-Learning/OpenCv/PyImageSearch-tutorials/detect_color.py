import argparse
import cv2 
import imutils
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Input image path")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])

# define the list of colors
boundaries = [
    ([17, 15, 100], [50, 56, 200]),     # Red
    ([86, 31, 4], [220, 88, 50]),       # Blue
    ([25, 146, 190], [62, 174, 250]),   # Yellow
    ([103, 86, 65], [145, 133, 128]),   # Gray
]

for lower, upper in boundaries:
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)

	# find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(img, lower, upper) # Pixels lying in this range will get 255 value and others 0 value
    # cv2.imshow("Mask", mask)
    output = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("images", np.hstack([img, output]))
    cv2.waitKey(0)