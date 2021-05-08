import numpy as np
import cv2
import math
from sklearn.feature_extraction import image
import pylab as pl


def enhance_eye_4_pupil(img_gray):
    normalizedImg = np.zeros((320, 240))
    normalized = cv2.normalize(img_gray, normalizedImg, 0, 255, cv2.NORM_MINMAX)

    _, threshold = cv2.threshold(normalized, 15, 255, cv2.THRESH_BINARY_INV)

    kernel_open = np.ones((7, 7), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)
    kernel_dilate = np.ones((7, 7), np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel_open)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close)
    return cv2.morphologyEx(closing, cv2.MORPH_DILATE, kernel_dilate)


def iris_circles(im_iris, radius_pupil):
    mean_shift = cv2.pyrMeanShiftFiltering(im_iris, 20, 30)
    gray_mean = cv2.cvtColor(mean_shift, cv2.COLOR_BGR2GRAY)
    return cv2.HoughCircles(gray_mean, cv2.HOUGH_GRADIENT, 1, 2 * radius_pupil, param1=100, param2=50,
                            minRadius=radius_pupil)


###################################
# Transform circle
def daugman_normalizaiton(im, height, width, r_in, r_out):  # Daugman归一化，输入为640*480,输出为width*height
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)  # Theta values
    r_out = r_in + r_out
    # Create empty flatten image
    flat = np.zeros((height, width, 3), np.uint8)
    circle_x = int(im.shape[0] / 2)
    circle_y = int(im.shape[1] / 2)

    for x in range(width):
        for j in range(height):
            theta = thetas[x]  # value of theta coordinate
            r_pro = j / height  # value of r coordinate(normalized)

            # get coordinate of boundaries
            Xi = circle_x + r_in * np.cos(theta)
            Yi = circle_y + r_in * np.sin(theta)
            Xo = circle_x + r_out * np.cos(theta)
            Yo = circle_y + r_out * np.sin(theta)

            # the matched cartesian coordinates for the polar coordinates
            Xc = (1 - r_pro) * Xi + r_pro * Xo
            Yc = (1 - r_pro) * Yi + r_pro * Yo

            color = im[int(Xc * .7)][int(Yc * .7)]  # color of the pixel, CHECK BUG if the value of color is high

            flat[j][x] = color
    return flat  # liang


def crop_eye_4_transform(im, x_I, y_I, r_I):
    if r_I > y_I:
        h_iris_begin = 0
    else:
        h_iris_begin = y_I - r_I

    h_iris_end = y_I + r_I

    if r_I > x_I:
        w_iris_begin = 0
    else:
        w_iris_begin = x_I - r_I

    w_iris_end = x_I + r_I

    return im[h_iris_begin:h_iris_end, w_iris_begin:w_iris_end]


def build_filters():
    filters = []
    ksize = 9   # 8×4 = 32 filters per images
    for theta in np.arange(0, np.pi, np.pi / 8):    # 8 ORIENTATIONS
        for lamda in np.arange(0, np.pi, np.pi / 4):    # 4 FREQUENCIES
            kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters


def main():
    global r_pupil, cX_pupil, cY_pupil, pupil, x_iris, r_iris, y_iris, pupil_existence
    path_read = '/media/manuel/HD_MOJEDA/DISCO_DURO/Mixto/Subjects/Image_Analysis/Project/dataset/images/055/'
    # path_write = '/media/manuel/HD_MOJEDA/DISCO_DURO/Mixto/Subjects/Image_Analysis/Project/Code/Images/'

    img_original = cv2.imread(path_read + '07_R.bmp')
    image_copy = img_original.copy()
    gray_img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    ##############################
    # Obtaining pupil
    dilate = enhance_eye_4_pupil(gray_img)
    contours_pupil, h = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for n in contours_pupil:
        a = cv2.contourArea(n)
        p = cv2.arcLength(n, True)
        circularity = (4 * math.pi * a) / (p * p)
        pupil_existence = False
        if circularity > 0.8 and a > 1000:
            pupil = n
            M = cv2.moments(n)
            cX_pupil = int(M["m10"] / M["m00"])
            cY_pupil = int(M["m01"] / M["m00"])
            r_pupil = int(p / (2 * math.pi))
            cv2.drawContours(img_original, n, -1, (0, 255, 0), 1)
            cv2.circle(img_original, (cX_pupil, cY_pupil), 2, (0, 255, 0), -1)
            pupil_existence = True
    if pupil_existence is False:  # In case the algorithm doesn't find any pupil at all
        cX_pupil = 160
        cY_pupil = 120
        r_pupil = 50

    ####################
    # Obtaining iris
    iris = iris_circles(image_copy, r_pupil)

    if iris is None:  # In case the algorithm doesn't find any circles at all
        x_iris = cX_pupil
        y_iris = cY_pupil
        r_iris = 2 * r_pupil
    else:
        iris_existence = False
        iris = np.uint16(np.around(iris))
        for i in iris[0, :]:
            if cv2.pointPolygonTest(pupil, (i[0], i[1]), True) >= 0:
                cv2.circle(img_original, (i[0], i[1]), i[2], (255, 0, 255), 2)
                cv2.circle(img_original, (i[0], i[1]), 2, (255, 0, 255), 3)
                # a_iris = round(math.pi * i[2] * i[2])
                x_iris, y_iris, r_iris = i
                iris_existence = True
        if iris_existence is False:  # In case the algorithm doesn't find any circle with a center within the pupil
            x_iris, y_iris, r_iris = iris[0][0]
            cv2.circle(img_original, (x_iris, y_iris), r_iris, (255, 0, 255), 2)
            cv2.circle(img_original, (x_iris, y_iris), 2, (255, 0, 255), 3)

    #######################
    # Obtaining flat image
    crop_im_eye = crop_eye_4_transform(image_copy, x_iris, y_iris, r_iris)
    flat = daugman_normalizaiton(crop_im_eye, (r_iris - r_pupil), int((2 * math.pi * (r_iris - r_pupil))), r_iris,
                                 r_pupil)
    resized_flat = cv2.resize(flat, (380, 60))
    print(type(flat), flat.shape)
    print(type(resized_flat), resized_flat.shape)

    ######################
    # Trying to get the features extraction
    patches = image.extract_patches_2d(resized_flat, (2, 2))
    print(type(patches), patches.shape)
    print(patches[1])

    ############################
    # Show of images
    cv2.imshow('img_original', img_original)
    cv2.imshow('resized_flat', resized_flat)

    cv2.waitKey(0)

    ###########################
    # Code for generating Garbor filters
    filters = build_filters()
    f = np.asarray(filters)
    print('Gabor Filters', f.shape)
    # output = np.asarray(res)
    # label = np.asarray(label)
    # print('Final output X,y', output.shape, label.shape)

    for k, im in enumerate(f[:32, :]):
        pl.subplot(6, 6, k + 1)
        pl.imshow(im.reshape(9, 9), cmap='gray')

    pl.show()


if __name__ == '__main__':
    main()
