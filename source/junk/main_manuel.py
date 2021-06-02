# import skimage
# from PIL import Image
import numpy as np
import cv2


def contrast_adaptive(img1):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img1)


def k_means_seg(img):
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2


# def nothing(x):
#     pass
kern_radius = 5
kernel = np.ones((kern_radius, kern_radius), np.uint8)

path_read = '/media/manuel/HD_MOJEDA/DISCO_DURO/Mixto/Subjects/Image_Analysis/Project/dataset/images/001/'
path_write = '/media/manuel/HD_MOJEDA/DISCO_DURO/Mixto/Subjects/Image_Analysis/Project/Code/Images/'

img_original = cv2.imread(path_read + '01_L.bmp')
# gray_img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
closing = cv2.morphologyEx(img_original, cv2.MORPH_CLOSE, kernel, iterations=2)
# cv2.imshow('closing', closing)
#
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)
# cv2.imshow('opening', opening)

gray_img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gray_img, 5)
pupil = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=50, minRadius=20, maxRadius=120)
print(pupil)
pupil = np.uint16(np.around(pupil))

# Mean shift filtering
mean_shift = cv2.pyrMeanShiftFiltering(img_original, 25, 30)

# mean_shift = cv2.pyrMeanShiftFiltering(opening, 25, 30)
# res = k_means_seg(mean_shift)

cv2.imshow('k-means', mean_shift)
gray_mean = cv2.cvtColor(mean_shift, cv2.COLOR_BGR2GRAY)

iris = cv2.HoughCircles(gray_mean, cv2.HOUGH_GRADIENT, 1, 400, param1=100, param2=50)
print(iris)
iris = np.uint16(np.around(iris))
# cv2.imshow('before contrast adaptive', mean_shift)
# cl1 = contrast_adaptive(gray_mean)
# cv2.imshow('after contrast adaptive', cl1)
# normalize_img = cv2.normalize(mean_shift, mean_shift, 0, 255, norm_type=cv2.NORM_MINMAX)
# cv2.imshow('Result', normalize_img)
# gray_mean = cv2.cvtColor(cl1, cv2.COLOR_BGR2GRAY)
# th1, threshold = cv2.threshold(gray_mean, 0, 180, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

for i in iris[0, :]:
    # draw the outer circle
    cv2.circle(img_original, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img_original, (i[0], i[1]), 2, (0, 0, 255), 3)

for i in pupil[0, :]:
    # draw the outer circle
    cv2.circle(img_original, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img_original, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('detected circles', img_original)
# cv2.imwrite(path_write + 'final_favorite.bmp', img_original)

# Canny
# cv2.namedWindow('image edges')
#
# cv2.createTrackbar('low_threshold', 'image edges', 800, 1300, nothing)
# cv2.createTrackbar('high_threshold', 'image edges', 1000, 1500, nothing)
#
# create switch for ON/OFF functionality
# switch = '0 : OFF \n1 : ON'
# cv2.createTrackbar(switch, 'image edges', 0, 1, nothing)
#
# edges = cv2.Canny(gray_img, 80, 150)
#
# while True:
#     cv2.imshow('image edges', edges)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
#
#     low_threshold = cv2.getTrackbarPos('low_threshold', 'image edges')
#     high_threshold = cv2.getTrackbarPos('high_threshold', 'image edges')
#     s = cv2.getTrackbarPos(switch, 'image edges')
#
#     edges = cv2.Canny(img_original, high_threshold, low_threshold)  # 500, 1400
#
#     if s == 1:
#         break

cv2.waitKey(0)
