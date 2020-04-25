import cv2
import imutils
from loader import display_images

def get_edges(img, display = False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 255)

    if display:
        display_images([('Edged', edged)])
    
    return edged

def findContours(img):
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        raise "No contour detected"

    return cnts

def drawContours(img, cnts, color=(0, 255, 0), thickness=2, display = False):
    cv2.drawContours(img, cnts, -1, color, thickness=thickness)
    if display:
        display_images([('Contoured', img)])