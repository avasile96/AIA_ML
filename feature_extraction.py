import numpy as np
import pandas as pd
# import math
from skimage.feature import local_binary_pattern
from skimage.measure import block_reduce
import pylab as pl
from timeit import default_timer as timer
# import glob
import cv2


def build_filters():
    filters = []
    ksize = 9
    for theta in np.arange(0, np.pi, np.pi / 8):  # 8 ORIENTATIONS
        for lamda in np.arange(0, np.pi, np.pi / 4):  # 4 FREQUENCIES, 32 FILTERS IN TOTAL
            kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
        np.maximum(accum, fimg, accum)
    return accum


def black_images_delete(result):
    result_gabor = []
    for n in range(len(result)):
        if n % 4 != 0:
            result_gabor.append(result[n])
    return result_gabor


def lbp_process(result_gabor):
    concat_hist = []
    for gabor in result_gabor:
        lbp_result = local_binary_pattern(gabor, 8, 1, method='ror')
        histogram_lbp, _ = np.histogram(lbp_result, bins=256)
        concat_hist.append(histogram_lbp)
    return concat_hist


def main():
    global histogram_concat, df_final, path_write

    ###########################
    # Code for generating Gabor filters
    filters = build_filters()

    for x in range(1, 225):

        if 1 <= x < 10:
            folder = '00' + str(x)
        elif 10 <= x <= 99:
            folder = '0' + str(x)
        else:
            folder = str(x)

        path_read_ubuntu = '/media/manuel/HD_MOJEDA/DISCO_DURO/Mixto/Subjects/Image_Analysis/Project/dataset/Strip/' \
                           'strip_' + folder + '_'
        # path_read_windows = 'F:\DISCO_DURO\Mixto\Subjects\Image_Analysis\Project\dataset\Strip\strip_' + folder + '_'
        # path_ground = '/media/manuel/HD_MOJEDA/DISCO_DURO/Mixto/Subjects/Image_Analysis/Project/' \
        #               'dataset/groundtruth/OperatorA_'
        path_write = '/media/manuel/HD_MOJEDA/DISCO_DURO/Mixto/Subjects/Image_Analysis/Project/Code/'
        res = []

        for patient in range(1, 11):

            strip = cv2.imread(path_read_ubuntu + str(patient) + '.bmp', 0)

            filters = np.asarray(filters)
            for i in range(len(filters)):
                res1 = process(strip, filters[i])
                res.append(np.asarray(res1))

            gabor_res = black_images_delete(res)

            histogram_concat = []
            histogram_concat = lbp_process(gabor_res)

        histograms = []
        cols = []
        for num_image in range(len(histogram_concat)):
            for value in range(len(histogram_concat[num_image])):
                histograms.append(histogram_concat[num_image][value])
                column = str(num_image) + '_' + str(value)
                cols.append(column)

        if x == 1:
            df = pd.DataFrame([histograms], index=[x], columns=cols)
            df_final = pd.DataFrame(df)
        else:
            df2 = pd.DataFrame([histograms], index=[x], columns=cols)
            df_final = df_final.append(df2, sort=False, ignore_index=False)
        print(f'The patient number {x} has been processed.')

    print('The process of saving the dataframe is starting...')
    df_final.to_csv(path_write + 'gabor32_8lbp1.csv')
    print('The dataframe has been saved.')


if __name__ == '__main__':
    start = timer()
    main()
    print("Time taken by the algorithm:", timer()-start)
