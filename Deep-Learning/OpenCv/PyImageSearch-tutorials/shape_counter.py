import argparse
import cv2
import imutils

# Setup command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help="Input path of the image")
ap.add_argument('-o', '--output', required=True, help="Output path of the image")
args = vars(ap.parse_args())

# Load the input image from disk
img = cv2.imread(args["input"])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
g_blur = cv2.GaussianBlur(gray, ksize=(5,5), sigmaX=0)
thresh = cv2.threshold(g_blur,60, 255, cv2.THRESH_BINARY)[1]


cv2.imshow("Original", img)
cv2.imshow("Blurred", g_blur)
cv2.imshow("Threshold", thresh)

cv2.waitKey(0)

# Extract contours from the image
cnts = cv2.findContours(thresh.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours and draw them on the input image
for c in cnts:
    cv2.drawContours(img, [c], -1, (0, 0, 255), thickness=2)

# Display the total number of shapes on the image
text = f"Total shapes: {len(cnts)}"
cv2.putText(img, text, (10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255), thickness=2)

# Write image to Output path on disk
cv2.imwrite(args["output"], img)