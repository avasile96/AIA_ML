import numpy as np
# import math
# from sklearn.feature_extraction import image
# from skimage.transform import integral_image
import pylab as pl
# from timeit import default_timer as timer
# import glob
import cv2


def build_filters():
    filters = []
    ksize = 9  # 8×4 = 32 filters per images
    for theta in np.arange(0, np.pi, np.pi / 8):  # 8 ORIENTATIONS
        for lamda in np.arange(0, np.pi, np.pi / 4):  # 4 FREQUENCIES
            kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters


def main():
    global r_pupil, cX_pupil, cY_pupil, pupil, x_iris, r_iris, y_iris, pupil_existence
    mean_lst = []
    num_folders = 0
    for x in range(10, 12):

        if 1 <= x < 10:
            folder = '00' + str(x)
        elif 10 <= x <= 99:
            folder = '0' + str(x)
        else:
            folder = str(x)

        path_read = '/media/manuel/HD_MOJEDA/DISCO_DURO/Mixto/Subjects/Image_Analysis/Project/Code/Strip/strip_' + \
                    folder + '_'
        # path_ground = '/media/manuel/HD_MOJEDA/DISCO_DURO/Mixto/Subjects/Image_Analysis/Project/' \
        #               'dataset/groundtruth/OperatorA_'
        # path_write = '/media/manuel/HD_MOJEDA/DISCO_DURO/Mixto/Subjects/Image_Analysis/Project/Code/Strip/'

        for patient in range(1, 11):

            strip = cv2.imread(path_read + str(patient) + '.bmp')

            ######################
            # Trying to get the features extraction
            # patches = image.extract_patches_2d(normalized_flat, (2, 2))
            # print(type(patches), patches.shape)
            # print(patches[1])
            # ii = integral_image(normalized_flat)

            ###########################
            # Code for generating Gabor filters
            filters = build_filters()
            f = np.asarray(filters)
            # print('Gabor Filters', f.shape)
            # output = np.asarray(res)
            # label = np.asarray(label)
            # print('Final output X,y', output.shape, label.shape)

            ############################
            # SHOW IMAGES
            cv2.imshow('Strip_' + folder + '_' + str(patient), strip)

    for k, im in enumerate(f[:32, :]):
        pl.subplot(6, 6, k + 1)
        pl.imshow(im.reshape(9, 9), cmap='gray')

    pl.show()

    cv2.waitKey(0)


if __name__ == '__main__':
    # start = timer()
    main()
    # print("Time taken by the algorithm:", timer()-start)
