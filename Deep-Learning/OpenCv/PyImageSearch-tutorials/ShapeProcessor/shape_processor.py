import cv2

class ShapeProcessor:
    def __init__(self):
        pass

    def detect(self, c):
        shape = "N/A"

        peri = cv2.arcLength(c, closed=True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, closed=True)

        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            # Can be Square or rectangle

            # compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
            shape = "Square" if ar >= 0.95 and ar <= 1.05 else "Rectangle"
        elif len(approx) == 5:
            shape = "Pentagon"
        elif len(approx) == 6:
            shape = "Hexagon"
        else:
            shape = "Circle"

        return shape
    
    def findCenter(self, c, ratio=1.0):
        M = cv2.moments(c)

        cX = int((M["m10"]/M["m00"])*ratio)
        cY = int((M["m01"]/M["m00"])*ratio)

        return cX, cY
    
