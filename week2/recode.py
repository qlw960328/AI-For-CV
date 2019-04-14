import cv2
import numpy as np

img = cv2.imread('lena.jpg')
cv2.imshow('lena', img)

# Gaussian kernel effect

img_gau = cv2.GaussianBlur(img, (7,7), 4)
cv2.imshow('gaussian img', img_gau)
cv2.waitKey()

# change the gaussian kernel size to change the image scale

mg_gau1 = cv2.GaussianBlur(img, (9,9), 4)
cv2.imshow('gaussian img1', img_gau1)
cv2.waitKey()

# change variance to change image scale

img_gau2 = cv2.GaussianBlur(img, (7,7), 0.2)
cv2.imshow('gaussian img2', img_gau2)
cv2.waitKey()

# use different method to get gausssian image

kernel = cv2.getGaussianKernel(7, 5) # kernel size variance
print(kernel)

g1_img = cv2.GaussianBlur(img, (7,7), 5)
g2_img = cv2.sepFilter2D(img, -1, kernel, kernel)
cv2.imshow('g1_img', g1_img)
cv2.imshow('g2_img', g2_img)
cv2.waitKey()

# 2nd derivative(边缘检测)

kernel_lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
lap_img = cv2.filter2D(img, -1, kernel=kernel_lap)
cv2.imshow('edge detection', lap_img)
cv2.waitKey()

# sharpen(图像锐化)

kernel_sharp = np.array([[0, 1, 0], [1, -3, 1], [0, 1, 0]], np.float32)
sharp_img = cv2.filter2D(img, -1, kernel=kernel_sharp)
cv2.imshow('sharp_image', sharp_img)
cv2.waitKey()

# 这样做因为周围有四个1，对该像素点起到一个平滑对作用，所以只有在边缘处有锐化对效果，其他部分应旧模糊

kernel_sharp = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]], np.float32)
sharp_img = cv2.filter2D(img, -1, kernel=kernel_sharp)
cv2.imshow('sharp_image', sharp_img)
cv2.waitKey()

# 更强对边缘效果，除了考虑x轴和y轴方向对梯度，同时考虑对角线方向对梯度

kernel_sharp = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)
sharp_img = cv2.filter2D(img, -1, kernel=kernel_sharp)
cv2.imshow('sharp_image', sharp_img)
cv2.waitKey()

# y轴 edge效果

edgey = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
edgey_img = cv2.filter2D(img, -1, kernel=edgey)
cv2.imshow('edgey', edgey_img)
cv2.waitKey()

# x轴edge效果

edgex = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
edgex_img = cv2.filter2D(img, -1, kernel=edgex)
cv2.imshow('edgex', edgex_img)
cv2.waitKey()

# find corner points in an image

img1 = cv2.imread('lena.jpg')
gray_img = np.float32(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))

img_harris = cv2.cornerHarris(gray_img, 2, 3, 0.05)
img_harris = cv2.dilate(img_harris, None)
cv2.imshow('corner harris', img_harris)
cv2.waitKey()

thorehold = 0.05 * np.max(img_harris)
img1[img_harris > thorehold] = [0, 0, 255]

cv2.imshow('corner harris img', img1)
cv2.waitKey()

# SIFT

sift = cv2.xfeatures2d.SIFT_create()
# detect SIFT
kp = sift.detect(img, None)
kp, des = sift.compute(img, kp)
print(des.shape)
img_sift= cv2.drawKeypoints(img,kp,outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('lenna_sift.jpg', img_sift)
key = cv2.waitKey()