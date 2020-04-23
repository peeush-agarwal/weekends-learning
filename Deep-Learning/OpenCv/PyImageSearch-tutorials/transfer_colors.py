import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Source image from which color needs to be transferred")
ap.add_argument("-t", "--target", required=True, help="Target image on which colors are applied")
ap.add_argument("-o", "--output", required=True, help="Path of output image")
args = vars(ap.parse_args())

def get_lab(bgr_img):
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype("float32")
    # cv2.imshow("Images", np.hstack([bgr_img, lab_img]))
    l, a, b = cv2.split(lab_img)
    # print(l, a, b)
    return l, a, b
    
def get_lab_mean_std(bgr_img):
    l, a, b = get_lab(bgr_img)

    l_mean = np.mean(l)
    a_mean = np.mean(a)
    b_mean = np.mean(b)

    l_sd = np.std(l)
    a_sd = np.std(a)
    b_sd = np.std(b)

    return (l_mean, l_sd), (a_mean, a_sd), (b_mean, b_sd)

source_img = cv2.imread(args["source"])
target_img = cv2.imread(args["target"])

(s_l_mean, s_l_sd), (s_a_mean, s_a_sd), (s_b_mean, s_b_sd) = get_lab_mean_std(source_img)
(t_l_mean, t_l_sd), (t_a_mean, t_a_sd), (t_b_mean, t_b_sd) = get_lab_mean_std(target_img)

l, a, b = get_lab(target_img)

l -= t_l_mean
a -= t_a_mean
b -= t_b_mean

l *= (s_l_sd / t_l_sd)
a *= (s_a_sd / t_a_sd)
b *= (s_b_sd / t_b_sd)

# l *= (t_l_sd / s_l_sd)
# a *= (t_a_sd / s_a_sd)
# b *= (t_b_sd / s_b_sd)

l += s_l_mean
a += s_a_mean
b += s_b_mean

l = np.clip(l, 0, 255)
a = np.clip(a, 0, 255)
b = np.clip(b, 0, 255)

target_img_lab = cv2.merge([l, a, b]).astype("uint8")
output = cv2.cvtColor(target_img_lab, cv2.COLOR_LAB2BGR)
cv2.imwrite(args["output"], output)

cv2.imshow('Output', output)
cv2.waitKey(0)
