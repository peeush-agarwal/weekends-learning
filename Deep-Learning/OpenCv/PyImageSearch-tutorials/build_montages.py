import argparse
import cv2
from imutils import build_montages
from imutils import paths
import random

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Path to input images")
ap.add_argument("-n", "--num_of_images", type=int, default=21, help="number of images to be montaged together")
args = vars(ap.parse_args())

image_paths = list(paths.list_images(args["images"]))
random.shuffle(image_paths)
image_paths = image_paths[:args["num_of_images"]]

images = []
for img_path in image_paths:
    images.append(cv2.imread(img_path))

montages = build_montages(images, (128, 196), (7,3))
for montage in montages:
    cv2.imshow("Montage", montage)
    cv2.waitKey(0)
