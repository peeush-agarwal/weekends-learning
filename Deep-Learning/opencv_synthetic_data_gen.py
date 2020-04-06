import cv2
import numpy as np
import random

height, width = (500, 500)

blank_img = np.zeros((height, width, 3), np.uint8)

channel = 0 #1
random_range = [0,150] #[0, 255]
for row in blank_img[:,:,channel]:
    for element in range(len(row)):
        row[element] = random.choice(random_range)

# cv2.imshow('Blank img', blank_img)

random_row = random.randint(0,100)
distractor_thickness = random.randint(10, 50)
distractor_width = random.randint(260, 400)

random_color = [random.randint(100,255),random.randint(100,255),random.randint(100,255)]
blank_img[random_row:(random_row+distractor_thickness),random_row:(random_row+distractor_width)] = random_color # [255,0,255]

# generate scratches
num_scratches = 3

for _ in range(random.randint(0,num_scratches)):
    row = random.randint(150, 250)
    blank_img[row:(row+random.randint(2,5)),row:(row+random.randint(35,70))] = [192,192,192]

non_glare = blank_img
cv2.imshow('Non-glared img', non_glare)

# add glare
image_lab = cv2.cvtColor(blank_img, cv2.COLOR_BGR2LAB)
l_channel,a,b = cv2.split(image_lab)

clahe = cv2.createCLAHE()
cl = clahe.apply(l_channel)
merge_channels = cv2.merge((cl,a,b))
glare_img = cv2.cvtColor(merge_channels, cv2.COLOR_LAB2BGR)

cv2.imshow('Glared img', glare_img)
cv2.waitKey()