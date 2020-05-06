import cv2
import imutils
import os

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
    file_name = os.path.basename(output_path)
    dir_path = output_path.replace(file_name, '')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok = True)
    cv2.imwrite(output_path, img)
