import cv2
import numpy as np
import math

def median_blur(img, kernel, padding_way):
    s = list(kernel)
    m, n = s[0], s[1]
    h = img.shape[0]
    l = img.shape[1]
    img_new = img
    if padding_way == 'ZERO':
        img = np.array(img)
        img = np.pad(img, ((math.ceil((m-1)/2.0),math.floor((n-1)/2.0)), (math.ceil((m-1)/2.0),math.floor((n-1)/2.0))), 'constant')
    if padding_way == 'REPLICA':
        img = np.array(img)
        img = np.pad(img, ((math.ceil((m-1)/2.0),math.floor((n-1)/2.0)), (math.ceil((m-1)/2.0),math.floor((n-1)/2.0))), 'edge')
    W = img.shape[0]
    H = img.shape[1]
    i = 0

    while i <= W-m:
        j = 0
        while j <= H-n:

            img_new[i][j] = np.median(img[i:m-1+i, j:n-1+j]).astype(img.dtype)

            j = j + 1
        i = i + 1
    return img_new

img = cv2.imread('lena.jpg', 0)
print(img)
cv2.imshow('img', img)
#img = img[100:200, 100:200]
img_blur = median_blur(img, (10,10), 'ZERO')
print(img_blur)
cv2.imshow('img_blur', img_blur)
cv2.waitKey()



