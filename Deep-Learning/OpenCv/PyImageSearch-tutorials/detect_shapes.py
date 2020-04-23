import argparse
import cv2
import imutils
from ImageProcessor.shape_detector import ShapeDetector
from ImageProcessor.shape_center import ShapeCenter
from ImageProcessor.color_labeler import ColorLabeler

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Input image path")
ap.add_argument('-w', '--width', type=int, default=300, help="Resize to width")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
resized = imutils.resize(img, width=args["width"])
ratio = img.shape[0] / float(resized.shape[0])

blurred = cv2.GaussianBlur(resized, (5, 5), 0)

gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

shape_detector = ShapeDetector()
shape_center = ShapeCenter()
color_labeler = ColorLabeler()

for c in cnts:
    cX, cY = shape_center.findCenter(c, ratio)

    shape = shape_detector.detect(c)
    color = color_labeler.label(lab, c)

    c = c.astype("float")
    c *= ratio
    c = c.astype("int")

    cv2.drawContours(img, [c], -1, (0, 255, 0), thickness=2)
    cv2.circle(img, (cX, cY), 7, (255, 255, 255), thickness=-1)
    cv2.putText(img, f'{color} {shape}', (cX-20, cY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)

    cv2.imshow('Shape', img)
    cv2.waitKey(500)

cv2.destroyAllWindows()