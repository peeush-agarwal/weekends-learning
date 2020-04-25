import cv2
import imutils

def load_image(path):
    return cv2.imread(path)

def resize_image(img, width = None, height = None):
    resized = None
    if not width is None:
        resized = imutils.resize(img, width = width)
    elif not height is None:
        resized = imutils.resize(img, height=height)
    else:
        raise f"Required one width or height"
    return resized

def display_images(images_with_titles, waitKey = 0):
    for title, img in images_with_titles:
        cv2.imshow(title, img)
    cv2.waitKey(waitKey)
    cv2.destroyAllWindows()

def write_to_disk(img, output_path):
    cv2.imwrite(output_path, img)