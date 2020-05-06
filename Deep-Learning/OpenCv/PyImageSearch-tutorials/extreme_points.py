import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Input image path")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
eroded = cv2.erode(gray, None, iterations=2)

thresh = cv2.threshold(eroded, 80, 255, cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

cnt = max(cnts, key=cv2.contourArea)

# print(cnt.shape)
left = cnt[:,:,0].argmin()
left = tuple(cnt[left][0])
# print(left)

right = cnt[:,:,0].argmax()
right = tuple(cnt[right][0])
# print(right)

bottom = cnt[:,:,1].argmax()
bottom = tuple(cnt[bottom][0])
# print(bottom)

top = cnt[:,:,1].argmin()
top = tuple(cnt[top][0])
# print(top)

cv2.drawContours(img, [cnt], -1, (0, 255, 0), thickness=2)
cv2.circle(img, left, 7, (0,0,255), thickness=-1)
cv2.circle(img, right, 7, (0,0,255), thickness=-1)
cv2.circle(img, bottom, 7, (0,0,255), thickness=-1)
cv2.circle(img, top, 7, (0,0,255), thickness=-1)

cv2.imshow('Contours', img)
cv2.waitKey(0)