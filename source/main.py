import numpy as np
import cv2
import math
import polarTransform


def contrast_adaptive(img1):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img1)


def k_means_seg(img):
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img.shape)
    return res2


def shadow_removal(meanshift):
    rgb_planes = cv2.split(meanshift)

    # result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    # result = cv2.merge(result_planes)
    return cv2.merge(result_norm_planes)


###################################
# Transform circle
def daugman_normalizaiton(image, height, width, r_in, r_out):       # Daugman归一化，输入为640*480,输出为width*height
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)  # Theta values
    r_out = r_in + r_out
    # Create empty flatten image
    flat = np.zeros((height,width, 3), np.uint8)
    circle_x = int(image.shape[0] / 2)
    circle_y = int(image.shape[1] / 2)

    for i in range(width):
        for j in range(height):
            theta = thetas[i]  # value of theta coordinate
            r_pro = j / height  # value of r coordinate(normalized)

            # get coordinate of boundaries
            Xi = circle_x + r_in * np.cos(theta)
            Yi = circle_y + r_in * np.sin(theta)
            Xo = circle_x + r_out * np.cos(theta)
            Yo = circle_y + r_out * np.sin(theta)

            # the matched cartesian coordinates for the polar coordinates
            Xc = (1 - r_pro) * Xi + r_pro * Xo
            Yc = (1 - r_pro) * Yi + r_pro * Yo

            color = image[int(Xc*.75)][int(Yc*.75)]  # color of the pixel

            flat[j][i] = color
    return flat  # liang

###################################


kern_radius = 5
kernel = np.ones((kern_radius, kern_radius), np.uint8)

path_read = '/media/manuel/HD_MOJEDA/DISCO_DURO/Mixto/Subjects/Image_Analysis/Project/dataset/images/218/'
path_write = '/media/manuel/HD_MOJEDA/DISCO_DURO/Mixto/Subjects/Image_Analysis/Project/Code/Images/'

img_original = cv2.imread(path_read + '05_L.bmp')
final_image = img_original.copy()
gray_img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

######################
# ALEX CODE
closing = cv2.morphologyEx(img_original, cv2.MORPH_CLOSE, kernel, iterations=2)
# cv2.imshow('closing', closing)
#
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)
# cv2.imshow('opening', opening)
#####################

img = cv2.medianBlur(gray_img, 5)
pupil = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=50, minRadius=20, maxRadius=120)
pupil = np.uint16(np.around(pupil))
print(pupil)

# Mean shift filtering
mean_shift = cv2.pyrMeanShiftFiltering(img_original, 25, 30)

#######################
# SHADOW REMOVAL
result_norm = shadow_removal(mean_shift)
# cv2.imshow('result shadow removal w norm', result_norm)
#######################

# cv2.imshow('mean shift filtering', mean_shift)
gray_mean = cv2.cvtColor(mean_shift, cv2.COLOR_BGR2GRAY)

iris = cv2.HoughCircles(gray_mean, cv2.HOUGH_GRADIENT, 1, 400, param1=100, param2=50)
iris = np.uint16(np.around(iris))
print(iris)

for i in iris[0, :]:
    # draw the outer circle
    cv2.circle(img_original, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img_original, (i[0], i[1]), 2, (0, 0, 255), 3)
    area_iris = round(math.pi * i[2] * i[2])

    print(area_iris)

for i in pupil[0, :]:
    # draw the outer circle
    iris_circle = cv2.circle(img_original, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img_original, (i[0], i[1]), 2, (0, 0, 255), 3)
    area_pupil = round(math.pi * i[2] * i[2])
    print(area_pupil)

cv2.imshow('detected circles', img_original)

w0, h0, _ = img_original.shape
x_iris, y_iris, r_iris = iris[0][0]
print(x_iris, y_iris, r_iris)
mask = np.zeros((w0, h0), dtype=np.uint8)
cv2.circle(mask, (x_iris, y_iris), r_iris, (255, 255, 255), -1, 8, 0)
out = gray_img * mask
white = 255 - mask
final = out + white

x_pupil, y_pupil, r_pupil = pupil[0][0]
print(x_pupil, y_pupil, r_pupil)
mask_iris = np.ones((w0, h0), dtype=np.uint8)
cv2.circle(mask_iris, (x_pupil, y_pupil), r_pupil, (0, 0, 0), -1)
out = final * mask_iris
white = 255 - mask_iris
final_iris = out + white
cv2.imshow('iris crop', final_iris)

inter_polar = cv2.warpPolar(final_iris, (round(r_iris), round(r_iris * math.pi)), (x_iris, y_iris),
                            r_iris, cv2.WARP_POLAR_LINEAR)

_, th1 = cv2.threshold(inter_polar, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
closing = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, (5, 5))
contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contour_iris = cv2.drawContours(inter_polar, contours, -1, (0, 255, 0), 1)

w, h = contour_iris.shape
mask_final = np.zeros((w, h), dtype=np.uint8)
cv2.drawContours(mask_final, contours, -1, (255, 255, 255), -1)
output = inter_polar * mask_final
white_output = 255 - mask_final
final_output = output + white_output
final1_output = cv2.bitwise_and(inter_polar, inter_polar, mask=mask_final)
final1_output = final1_output[:, int(r_pupil/2):]
cv2.imshow('final_output', final_output)
cv2.imshow('final1_output', final1_output)

# cv2.imshow('cartographic trans', inter_polar)
# cv2.imshow('binarization', closing)

# polarImage, ptSettings = polarTransform.convertToPolarImage(final_image, (x_pupil, y_pupil), r_pupil, r_iris,
#                                                             hasColor=True, order=0)
# cv2.imshow('polarImage', polarImage)
# print(polarImage.shape)
print(2 * math.pi * (r_iris - r_pupil))
print(r_iris - r_pupil)
flat = daugman_normalizaiton(final_image, (r_iris - r_pupil), int((2 * math.pi * (r_iris - r_pupil))), r_iris, r_pupil)
cv2.imshow('flat image', flat)
# cv2.imshow('original', img_original)
# cv2.imshow('mean shift', mean_shift)
# cv2.imshow('result shadow removal w/o norm', result)


cv2.waitKey(0)
