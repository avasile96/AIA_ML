
def stripTease(seg_img, center, max_radius): # TODO
    flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    final_strip = cv2.linearPolar(seg_img, center, max_radius, flags)
    return final_strip

def polar_transform(im_from_gen, pred_of_im):
    """
    val_gen.__getitem__(0)[0][0] DESCRIPTION
    ...
    First index represents the input/target pair selection (0 = first pair)
    Second index represents the input/target selection (0 is input)
    Thrid index is the n-th image in the minibatch (0 = first image)
    --> so in the line below we access the first image of the first
    input of the first input/target pair 
    ...
    I've thaught about the contour filling part, it doesn't matter which
    contour you fill, the pupil still gets filled
    """
    og_copy = og_image = im_from_gen # getting og image
    prediction = pred_of_im # getting prediction
    f3 = plt.figure()
    f3.suptitle('og_copy')
    io.imshow(og_copy) 
    f3 = plt.figure()
    f3.suptitle('prediction')
    io.imshow(prediction) 

    # tresholding
    th1, binary_pred = cv2.threshold(np.squeeze(prediction),0.1,1,cv2.THRESH_BINARY)
    binary_pred = np.uint8(binary_pred)
    f3 = plt.figure()
    f3.suptitle('binary_pred')
    io.imshow(binary_pred) 
    # opening
    kern_radius = 5
    kernel = np.ones((kern_radius,kern_radius),np.uint8)
    open_mask = cv2.morphologyEx(binary_pred, cv2.MORPH_OPEN, kernel, iterations = 2)
    f3 = plt.figure()
    f3.suptitle('open_mask')
    io.imshow(open_mask) 
    # multiplying threshold with og image
    cut_image = np.multiply(og_image, open_mask)
    f3 = plt.figure()
    f3.suptitle('cut_image')
    io.imshow(cut_image)
    
    # HOUGH CIRCLES: getting pupil circle (x_center, y_center, rad)
    # pred_sq = np.squeeze(prediction)*255
    # pred_sq_uint8 = np.uint8(pred_sq)
    # open_mask_blur = cv2.GaussianBlur(open_mask,(5,5),0)
    # pupil_outline = cv2.HoughCircles(open_mask_blur, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=50, minRadius=20, maxRadius=50)
    # if (pupil_outline is None):
    #     center = (np.floor(img_size[0]/2),np.floor(img_size[1]/2))
    # else:
    #     pupil_outline = np.uint16(np.around(pupil_outline))
    #     center = (np.squeeze(pupil_outline)[0], np.squeeze(pupil_outline)[1])
      
    cnt, hierarchy = cv2.findContours(open_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(og_copy, cnt,  -1, (255,0,0), 2)
    cv2.imshow('Objects Detected',og_copy)
    byakugan = cv2.fillPoly(open_mask, pts =cnt, color=(255,255,255))
    f4 = plt.figure()
    f4.suptitle('byakugan')
    io.imshow(byakugan)
    
    dist_im = cv2.distanceTransform(byakugan, cv2.DIST_L2, 3)
    cv2.normalize(dist_im, dist_im, 0, 1.0, cv2.NORM_MINMAX)
    f4 = plt.figure()
    f4.suptitle('distance map')
    io.imshow(dist_im)
    
    th2, centers = cv2.threshold(np.squeeze(dist_im),0.99,1.1,cv2.THRESH_BINARY)
    centers = np.uint8(centers)
    f3 = plt.figure()
    f3.suptitle('centers')
    io.imshow(centers)
    
    np.count_nonzero(centers==1)
    dist_centers = cv2.distanceTransform(centers, cv2.DIST_L1, 3)
    f3 = plt.figure()
    f3.suptitle('dist_centers')
    io.imshow(dist_centers)
    np.count_nonzero(dist_centers==1)
    
    centers_coordinates = []
    
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            if (centers[i,j] == 1):
                centers_coordinates.append((i,j))
    
    dist_plus_centers = np.add(dist_im, centers)
    f3 = plt.figure()
    f3.suptitle('dist_plus_centers')
    io.imshow(dist_plus_centers) 
    
    euc = []
    for i in cnt[0][:,:]:
        euc.append(distance.euclidean(i, center))
    strip = stripTease(cut_image, center, np.min(np.array(euc)))
    return strip


#%% DISTANCE MAP
    # Getting Contours
    cnt, hierarchy = cv2.findContours(open_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(og_copy, cnt,  -1, (255,0,0), 2)
    
    # Distance transform of Iris Interior
    dist_im = cv2.distanceTransform(byakugan, cv2.DIST_L2, 3)
    cv2.normalize(dist_im, dist_im, 0, 1.0, cv2.NORM_MINMAX)
    
    # Center Candidates Selection from Distance transform
    th2, centers = cv2.threshold(np.squeeze(dist_im),0.99,1.1,cv2.THRESH_BINARY)
    
    # Center Selection from Potential Candidates
    centers_coordinates = []
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            if (centers[i,j] == 1):
                centers_coordinates.append((i,j))
                
    centers_array = np.array(centers_coordinates)
    ideal_center = (np.mean(centers_array[:,0]), np.mean(centers_array[:,1]))
    distances_from_ideal = []
    for i in centers_coordinates:
        distances_from_ideal.append(distance.euclidean(i, ideal_center))
    
    distances_from_ideal_array = np.array(distances_from_ideal)
    min_distance_from_ideal = distances_from_ideal_array.min()
    min_distance_from_ideal_index = distances_from_ideal.index(min_distance_from_ideal)
    center = centers_coordinates[min_distance_from_ideal_index]
    
    # Testing center positioning
    prediction[center[0],center[1]] = 1
    f3 = plt.figure()
    f3.suptitle('prediction')
    io.imshow(prediction) 
    
    open_mask[center[0],center[1]] = 2
    f3 = plt.figure()
    f3.suptitle('open_mask')
    io.imshow(open_mask) 