import argparse
import cv2
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import os

import loader
import processor


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help = "Path of input image")
ap.add_argument('-o', '--output', required=True, help = "Path of output image")
ap.add_argument('-s', '--show_intermediate', type=bool, default=False, help="Show intermediate steps")
args = vars(ap.parse_args())

# define the answer key which maps the question number
# to the correct answer
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

# If display intermediate steps
display_intermediate = args["show_intermediate"]

img = loader.load_image(args["input"])
orig = img.copy()
img = loader.resize_image(img.copy(), height=400)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if display_intermediate:
    loader.display_images([('Original', orig), ('Resized', img)])

edged = processor.get_edges(img, display=display_intermediate)

cnts = processor.findContours(edged)

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
docCnt = None
for c in cnts:
    peri = cv2.arcLength(c, closed=True)
    approx = cv2.approxPolyDP(c, 0.04*peri, closed=True)

    if len(approx) == 4:
        docCnt = approx
        break

processor.drawContours(img.copy(), [docCnt], color=(0, 255, 0), display=display_intermediate)

transformed = four_point_transform(img, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4,2))

if display_intermediate:
    loader.display_images([('Transformed', transformed), ('Warped', warped)])

thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

if display_intermediate:
    loader.display_images([('Thresholded', thresh)])

cnts = processor.findContours(thresh.copy())

bubbleCnts = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h)

    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        bubbleCnts.append(c)

processor.drawContours(transformed.copy(), bubbleCnts, color=(0, 0, 255), display=display_intermediate)

questionCnts = contours.sort_contours(bubbleCnts, method="top-to-bottom")[0]
correct = 0

for (quesID, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    rowCnts = contours.sort_contours(questionCnts[i:i+5])[0]

    # output = cv2.drawContours(transformed.copy(), list(rowCnts), -1, (0, 0, 255), thickness=2)
    # cv2.imshow('Output', output)
    # cv2.waitKey(0)

    bubbled = None
    for idx, rowCnt in enumerate(rowCnts):
        mask = np.zeros(thresh.shape, dtype="uint8")
        mask = cv2.drawContours(mask, [rowCnt], -1, 255, thickness=-1)
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        
        # cv2.imshow('Output', mask)
        # cv2.waitKey(1000)

        total = cv2.countNonZero(mask)
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, idx)
    
    # print(f'Question {quesID}: Bubbled {bubbled[1]}')
    color = (0, 0, 255)
    answer = ANSWER_KEY[quesID]

    if answer == bubbled[1]:
        color = (0, 255, 0)
        correct += 1
    
    cv2.drawContours(transformed, [rowCnts[answer]], -1, color, thickness=3)

percent = (correct / 5.0) * 100
cv2.putText(transformed, f'{percent:.2f} %', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), thickness=2)

loader.display_images([('Original', orig),('Final', transformed)])

loader.write_to_disk(transformed, args["output"])
