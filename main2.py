import numpy as np
import cv2
import math
# from sklearn.feature_extraction import image
# from skimage.transform import integral_image
# import pylab as pl
from timeit import default_timer as timer
import glob


def enhance_eye_4_pupil(img_gray):
    normalizedImg = np.zeros((320, 240))
    normalized = cv2.normalize(img_gray, normalizedImg, 0, 255, cv2.NORM_MINMAX)

    _, threshold = cv2.threshold(normalized, 15, 255, cv2.THRESH_BINARY_INV)

    kernel_open = np.ones((7, 7), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)
    kernel_dilate = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel_close)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)
    return cv2.morphologyEx(opening, cv2.MORPH_DILATE, kernel_dilate)


def pupil_contour_calculation_drawing(original_im, pupil_contour):
    global pupil_info, pupil_cX, pupil_cY, existence_pupil, pupil_r
    for n in pupil_contour:
        a = cv2.contourArea(n)
        p = cv2.arcLength(n, True)
        circularity = (4 * math.pi * a) / (p * p)
        existence_pupil = False
        # cv2.drawContours(original_im, n, -1, (255, 0, 0), 1)
        if circularity > 0.70 and a > 900:
            pupil_info = n
            M = cv2.moments(n)
            pupil_cX = int(M["m10"] / M["m00"])
            pupil_cY = int(M["m01"] / M["m00"])
            pupil_r = int(p / (2 * math.pi))
            cv2.drawContours(original_im, n, -1, (255, 0, 0), 1)
            cv2.circle(original_im, (pupil_cX, pupil_cY), 2, (0, 255, 0), -1)
            existence_pupil = True
    if existence_pupil is False:  # In case the algorithm doesn't find any pupil at all
        pupil_cX = 160
        pupil_cY = 120
        pupil_r = 50
        # pupil_info = None

    return pupil_info, pupil_cX, pupil_cY, pupil_r


def iris_circles(im_iris, radius_pupil):
    mean_shift = cv2.pyrMeanShiftFiltering(im_iris, 20, 30)
    gray_mean = cv2.cvtColor(mean_shift, cv2.COLOR_BGR2GRAY)
    gray_mean = cv2.medianBlur(gray_mean, 5)
    return cv2.HoughCircles(gray_mean, cv2.HOUGH_GRADIENT, 1.2, 2 * radius_pupil, param1=100, param2=30,
                            minRadius=int(5/3*radius_pupil))


def iris_contour_calculation_drawing(original_im, iris_contour, pupil_contour):
    global iris_x, iris_y, iris_r
    if iris_contour is None:  # In case the algorithm doesn't find any circles at all
        iris_x = cX_pupil
        iris_y = cY_pupil
        iris_r = 2 * r_pupil
        cv2.circle(original_im, (iris_x, iris_y), iris_r, (255, 0, 255), 2)
        cv2.circle(original_im, (iris_x, iris_y), 2, (255, 0, 255), 3)
    else:
        iris_existence = False
        iris = np.uint16(np.around(iris_contour))
        for i in iris[0, :]:
            if cv2.pointPolygonTest(pupil_contour, (i[0], i[1]), measureDist=False) > 0:
                cv2.circle(original_im, (i[0], i[1]), i[2], (255, 0, 255), 2)
                cv2.circle(original_im, (i[0], i[1]), 2, (255, 0, 255), 3)
                # a_iris = round(math.pi * i[2] * i[2])
                iris_x, iris_y, iris_r = i
                iris_existence = True
        if iris_existence is False:  # In case the algorithm doesn't find any circle with a center within the pupil
            iris_x = cX_pupil
            iris_y = cY_pupil
            iris_r = 2 * r_pupil
            cv2.circle(original_im, (iris_x, iris_y), iris_r, (255, 0, 255), 2)
            cv2.circle(original_im, (iris_x, iris_y), 2, (255, 0, 255), 3)

    return iris_x, iris_y, iris_r


def stripTease(seg_img, center, max_radius):  # TODO
    flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    final_strip = cv2.linearPolar(seg_img, center, max_radius, flags)
    return final_strip


def build_filters():
    filters = []
    ksize = 9  # 8×4 = 32 filters per images
    for theta in np.arange(0, np.pi, np.pi / 8):  # 8 ORIENTATIONS
        for lamda in np.arange(0, np.pi, np.pi / 4):  # 4 FREQUENCIES
            kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters


def performance_evaluation(iris_mask, im_ground):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(0, 240):
        for j in range(0, 320):
            if iris_mask[i][j] == im_ground[i][j] == 255:
                TP += 1
            if iris_mask[i][j] == 255 and iris_mask[i][j] != im_ground[i][j]:
                FP += 1
            if iris_mask[i][j] == im_ground[i][j] == 0:
                TN += 1
            if iris_mask[i][j] == 0 and iris_mask[i][j] != im_ground[i][j]:
                FN += 1
    return TP, FP, TN, FN


def calculation_performance(TP, FP, TN, FN):
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    return (2 * R * P) / (R + P)


def ground_truth_acquisition(ground_path, folder):
    ground_truth_lst = []
    for num in range(1, 11):
        if num < 6:
            ground_truth = cv2.imread(ground_path + folder + '-A_0' + str(num) + '.tiff', 0)
            ground_truth_lst.append(ground_truth)
        if 5 < num < 10:
            ground_truth = cv2.imread(ground_path + folder + '-B_0' + str(num) + '.tiff', 0)
            ground_truth_lst.append(ground_truth)
        if num == 10:
            ground_truth = cv2.imread(ground_path + folder + '-B_' + str(num) + '.tiff', 0)
            ground_truth_lst.append(ground_truth)
    return ground_truth_lst


def main():
    global r_pupil, cX_pupil, cY_pupil, pupil, x_iris, r_iris, y_iris, pupil_existence
    mean_lst = []
    num_folders = 0
    for x in range(1, 2):

        if 1 <= x < 10:
            folder = '00' + str(x)
        elif 10 <= x <= 99:
            folder = '0' + str(x)
        else:
            folder = str(x)

        path_read = '/media/manuel/HD_MOJEDA/DISCO_DURO/Mixto/Subjects/Image_Analysis/Project/dataset/images/' + \
                    folder + '/*.bmp'
        path_ground = '/media/manuel/HD_MOJEDA/DISCO_DURO/Mixto/Subjects/Image_Analysis/Project/' \
                      'dataset/groundtruth/OperatorA_'
        # path_write = '/media/manuel/HD_MOJEDA/DISCO_DURO/Mixto/Subjects/Image_Analysis/Project/Code/Strip/'
        picture = 1
        index = 0
        F_total = 0

        #######################
        # Acquisition of the ground truth images to evaluate
        ground_truth_lst = ground_truth_acquisition(path_ground, folder)

        for img in glob.glob(path_read):

            img_original = cv2.imread(img)
            image_copy = img_original.copy()
            gray_img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

            ##############################
            # Obtaining pupil
            dilate = enhance_eye_4_pupil(gray_img)
            contours_pupil, h = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pupil, cX_pupil, cY_pupil, r_pupil = pupil_contour_calculation_drawing(img_original, contours_pupil)

            ####################
            # Obtaining iris
            iris = iris_circles(img_original, r_pupil)
            x_iris, y_iris, r_iris = iris_contour_calculation_drawing(img_original, iris, pupil)

            #######################
            # Obtaining flat image and normalize
            strip = stripTease(image_copy, (x_iris, y_iris), r_iris)
            normalized_zeros = np.zeros((240, 320))
            normalized_strip = cv2.normalize(strip, normalized_zeros, 0, 255, cv2.NORM_MINMAX)

            ######################
            # Get final masks for evaluation
            mask_iris = np.zeros((240, 320))
            cv2.circle(mask_iris, (x_iris, y_iris), r_iris, 255, -1)
            cv2.drawContours(mask_iris, [pupil], -1, 0, cv2.FILLED)
            ground_im = ground_truth_lst[index]

            ######################
            # Performance evaluation
            TP, FP, TN, FN = performance_evaluation(mask_iris, ground_im)
            F1 = calculation_performance(TP, FP, TN, FN)
            print(f'image number: {index + 1}; F1: {F1}')
            F_total = F_total + F1

            ############################
            # SHOW IMAGES
            cv2.imshow('img_original_' + '_' + folder + '_' + str(picture), img_original)
            cv2.imshow('normalized_strip_' + folder + '_' + str(picture), normalized_strip)
            cv2.imshow('mask_iris_' + '_' + folder + '_' + str(picture), mask_iris)
            # cv2.imwrite(path_write + 'strip_' + folder + '_' + str(window) + '.bmp', normalized_strip)

            picture += 1
            index += 1

        mean = F_total / 10
        print(f'Folder number: ' + str(x) + ', with an average of: ', mean)
        mean_lst.append(mean)
        cv2.waitKey(0)
        num_folders += 1
        mean_total = np.mean(mean_lst)
        print('Average for ' + str(num_folders) + ' folders is: ', mean_total)


if __name__ == '__main__':
    start = timer()
    main()
    print("Time for 1 folder:", timer()-start)
