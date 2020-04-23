import cv2

class ShapeCenter:
    def __init__(self):
        pass

    def findCenter(self, c, ratio=1.0):
        M = cv2.moments(c)

        cX = int((M["m10"]/M["m00"])*ratio)
        cY = int((M["m01"]/M["m00"])*ratio)

        return cX, cY
