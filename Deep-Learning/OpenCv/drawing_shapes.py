import cv2
import numpy as np

blank_img = np.zeros((512, 512, 3), dtype=np.uint8)

# Draw a blue line, 5px thick from top-left to bottom-right 
img = cv2.line(blank_img, (0, 0), (511, 511), (255, 0, 0), thickness=5)

# Draw a rectangle in top-right region
img = cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), thickness=3)

# Draw a circle inside above rectangle
img = cv2.circle(img, (447, 63), 63, (0, 0, 255), thickness=-1)

# Draw ellipsis
# img = cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)
img = cv2.ellipse(img, (256, 256), (100, 50), angle=0, startAngle=45, endAngle=270, color=255, thickness=-1)

# Draw a polygon
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
img = cv2.polylines(img,[pts],True,(0,255,255))

# Put text on image
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img,'Diff. shapes',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

#cv2.imshow('Image', img)
#cv2.waitKey(0)

def draw_open_cv_logo():
    img = np.zeros((320, 320, 3), dtype=np.uint8)

    # Circle - O
    img = cv2.circle(img, (128, 64), 60, (0, 0, 255), -1)
    img = cv2.circle(img, (128, 64), 25, (0, 0, 0), -1)

    # Circle - C
    img = cv2.circle(img, (64, 192), 60, (0, 255, 0), -1)
    img = cv2.circle(img, (64, 192), 25, (0, 0, 0), -1)

    # Triangle
    pts = np.array([[128, 64], [64, 192], [192, 192]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    img = cv2.drawContours(img, [pts], 0, (0, 0, 0), thickness=-1)

    # Circle - V
    img = cv2.circle(img, (192, 192), 60, (255, 0, 0), -1)
    img = cv2.circle(img, (192, 192), 25, (0, 0, 0), -1)

    # Triangle
    pts = np.array([[192, 192],[160, 128], [224, 128]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    img = cv2.drawContours(img, [pts], 0, (0, 0, 0), thickness=-1)

    cv2.putText(img, 'OpenCV', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow('Logo', img)
    cv2.waitKey(0)

draw_open_cv_logo()
cv2.destroyAllWindows()
