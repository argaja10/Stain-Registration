# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 01:08:29 2024

@author: argajashende
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from visualization import plot_landmarks, blend_images, read_coordinates_from_csv, plot_landmarks_with_overlay
from PIL import Image


def SIFT_registration(source_image_path, normalized_target_image_path, target_image_path, target_landmarks):
    # Preprocessing images
    # Read the images in grayscle, downscale and apply median blur
    source_image = cv.imread(source_image_path, cv.IMREAD_GRAYSCALE) 
    target_image = cv.imread(normalized_target_image_path, cv.IMREAD_GRAYSCALE)
    source_image = cv.medianBlur(cv.resize(source_image, (0, 0), fx = 0.1, fy = 0.1),5)
    target_image = cv.medianBlur(cv.resize(target_image, (0, 0), fx = 0.1, fy = 0.1),5)
    
    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(target_image,None)
    kp2, des2 = sift.detectAndCompute(source_image,None)
    cv.setRNGSeed(2391)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 30)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
     if m.trainIdx != m.queryIdx and m.distance < 0.8*n.distance:
         good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
     #h,w = img1.shape
     #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
     #dst = cv.perspectiveTransform(pts,M)
     #img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
     
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
     singlePointColor = None,
     matchesMask = matchesMask, # draw only inliers
     flags = 2)
    matches_image = cv.drawMatches(target_image,kp1,source_image,kp2,good,None,**draw_params)
    plt.figure(figsize=(10,20), dpi=600)
    plt.imshow(matches_image, 'gray'),plt.show()


    # Extract location of matches
    points1 = np.float32([kp1[m.queryIdx].pt for m in good])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # Find homography
    homography_matrix, mask = cv.findHomography(points1, points2, cv.RANSAC, 5.0)

    # Register
    h, w= source_image.shape[:2]
    target_image = cv.imread(target_image_path)
    S = np.array([[10, 0, 0], [0, 10, 0], [0, 0 ,1]])
    Sinv = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]])
    homography_matrix_rescaled = np.matmul(S, homography_matrix)
    homography_matrix_rescaled = np.matmul(homography_matrix_rescaled, Sinv)
    registered_image = cv.warpPerspective(target_image, homography_matrix_rescaled, (10*w, 10*h))

    #cv.imwrite('registered_image-3.jpg',registered_image)
    #plt.imshow(registered_image)
    #plt.axis('off')
    
    # Reshape the landmarks to a format compatible with cv2.perspectiveTransform
    #target_landmarks = np.array(target_landmarks).reshape(-1, 1, 2)

    # Transform the coordinates
    transformed_landmarks = cv.perspectiveTransform(np.array(target_landmarks).reshape(-1, 1, 2), homography_matrix_rescaled)

    # Reshape back to original format
    transformed_landmarks = transformed_landmarks.reshape(-1, 2)
    transformed_landmarks_list = [tuple(np.round(coord,1)) for coord in transformed_landmarks]

    return registered_image, transformed_landmarks_list



# Case 1: Registration of CC10 to H&E stain
source_image_path = './TestData/01-HE.jpg'
source_image = np.array(Image.open(source_image_path))
source_landmarks_path = './TestData/01-HE.csv'
source_landmarks = read_coordinates_from_csv(source_landmarks_path)

target_image_path = './TestData/01-CC10.jpg'
target_landmarks_path = './TestData/01-CC10.csv'
target_landmarks = read_coordinates_from_csv(target_landmarks_path)

normalized_target_image_path = './TestData/normalized-01-CC10.jpg'

registered_image, transformed_landmarks_list = SIFT_registration(source_image_path, normalized_target_image_path, target_image_path, target_landmarks)

plt.imshow(registered_image)

plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks_list, label='Before Registration')
plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks_list, label='After Registration')


# Case 2: Registration of CD31 to H&E stain
source_image_path = './TestData/01-HE.jpg'
source_image = np.array(Image.open(source_image_path))
source_landmarks_path = './TestData/01-HE.csv'
source_landmarks = read_coordinates_from_csv(source_landmarks_path)

target_image_path = './TestData/01-CD31.jpg'
target_landmarks_path = './TestData/01-CD31.csv'
target_landmarks = read_coordinates_from_csv(target_landmarks_path)

normalized_target_image_path = './TestData/normalized-01-CD31.jpg'

registered_image, transformed_landmarks_list = SIFT_registration(source_image_path, normalized_target_image_path, target_image_path, target_landmarks)

plt.imshow(registered_image)

plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks_list, label='Before Registration')
plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks_list, label='After Registration')


# Redefine fuction to tune hyperparaeters
def SIFT_registration(source_image_path, normalized_target_image_path, target_image_path, target_landmarks):
    # Preprocessing images
    # Read the images in grayscle, downscale and apply median blur
    source_image = cv.imread(source_image_path, cv.IMREAD_GRAYSCALE) 
    target_image = cv.imread(normalized_target_image_path, cv.IMREAD_GRAYSCALE)
    source_image = cv.medianBlur(cv.resize(source_image, (0, 0), fx = 0.05, fy = 0.05),5)
    target_image = cv.medianBlur(cv.resize(target_image, (0, 0), fx = 0.05, fy = 0.05),7)
    
    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(target_image,None)
    kp2, des2 = sift.detectAndCompute(source_image,None)
    FLANN_INDEX_KDTREE = 1
    cv.setRNGSeed(2391)
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 60)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
     if m.trainIdx != m.queryIdx and m.distance < 0.85*n.distance:
         good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
     #h,w = img1.shape
     #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
     #dst = cv.perspectiveTransform(pts,M)
     #img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
     
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
     singlePointColor = None,
     matchesMask = matchesMask, # draw only inliers
     flags = 2)
    matches_image = cv.drawMatches(target_image,kp1,source_image,kp2,good,None,**draw_params)
    plt.figure(figsize=(10,20), dpi=600)
    plt.imshow(matches_image, 'gray'),plt.show()


    # Extract location of matches
    points1 = np.float32([kp1[m.queryIdx].pt for m in good])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # Find homography
    homography_matrix, mask = cv.findHomography(points1, points2, cv.RANSAC, 5.0)

    # Register
    h, w= source_image.shape[:2]
    target_image = cv.imread(target_image_path)
    S = np.array([[20, 0, 0], [0, 20, 0], [0, 0 ,1]])
    Sinv = np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 1]])
    homography_matrix_rescaled = np.matmul(S, homography_matrix)
    homography_matrix_rescaled = np.matmul(homography_matrix_rescaled, Sinv)
    registered_image = cv.warpPerspective(target_image, homography_matrix_rescaled, (20*w, 20*h))

    #cv.imwrite('registered_image-3.jpg',registered_image)
    #plt.imshow(registered_image)
    #plt.axis('off')
    
    # Reshape the landmarks to a format compatible with cv2.perspectiveTransform
    #target_landmarks = np.array(target_landmarks).reshape(-1, 1, 2)

    # Transform the coordinates
    transformed_landmarks = cv.perspectiveTransform(np.array(target_landmarks).reshape(-1, 1, 2), homography_matrix_rescaled)

    # Reshape back to original format
    transformed_landmarks = transformed_landmarks.reshape(-1, 2)
    transformed_landmarks_list = [tuple(np.round(coord,1)) for coord in transformed_landmarks]

    return registered_image, transformed_landmarks_list


# Case 3: Registration of Ki67 to H&E stain
source_image_path = './TestData/01-HE.jpg'
source_image = np.array(Image.open(source_image_path))
source_landmarks_path = './TestData/01-HE.csv'
source_landmarks = read_coordinates_from_csv(source_landmarks_path)

target_image_path = './TestData/01-Ki67.jpg'
target_landmarks_path = './TestData/01-Ki67.csv'
target_landmarks = read_coordinates_from_csv(target_landmarks_path)

normalized_target_image_path = './TestData/normalized-01-Ki67.jpg'

registered_image, transformed_landmarks_list = SIFT_registration(source_image_path, normalized_target_image_path, target_image_path, target_landmarks)

plt.imshow(registered_image)

plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks_list, label='Before Registration')
plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks_list, label='After Registration')




# source_image = cv.imread(source_image_path, cv.IMREAD_GRAYSCALE) 
# target_image = cv.imread(normalized_target_image_path, cv.IMREAD_GRAYSCALE)
# source_image = cv.medianBlur(cv.resize(source_image, (0, 0), fx = 0.08, fy = 0.08),5)
# target_image = cv.medianBlur(cv.resize(target_image, (0, 0), fx = 0.08, fy = 0.08),7)

# plt.imshow(target_image, cmap='gray')

# plt.imshow(source_image, cmap='gray')




# MIN_MATCH_COUNT = 10
# img1 = cv.imread('./TestData/normalized.jpg', cv.IMREAD_GRAYSCALE) # queryImage
# img2 = cv.imread('./TestData/01-HE.jpg', cv.IMREAD_GRAYSCALE) # trainImage


# img1 = cv.medianBlur(cv.resize(img1, (0, 0), fx = 0.1, fy = 0.1),5)
# img2 = cv.medianBlur(cv.resize(img2, (0, 0), fx = 0.1, fy = 0.1),5)

# # Initiate SIFT detector
# sift = cv.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 30)
# flann = cv.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1,des2,k=2)
# # store all the good matches as per Lowe's ratio test.
# good = []
# for m,n in matches:
#  if m.trainIdx != m.queryIdx and m.distance < 0.8*n.distance:
#      good.append(m)

# if len(good)>MIN_MATCH_COUNT:
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
#     matchesMask = mask.ravel().tolist()
#  #h,w = img1.shape
#  #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#  #dst = cv.perspectiveTransform(pts,M)
#  #img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
# else:
#     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
#     matchesMask = None
 
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#  singlePointColor = None,
#  matchesMask = matchesMask, # draw only inliers
#  flags = 2)
# img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
# plt.figure(figsize=(10,20), dpi=600)
# plt.imshow(img3, 'gray'),plt.show()


# # Extract location of matches
# points1 = np.float32([kp1[m.queryIdx].pt for m in good])
# points2 = np.float32([kp2[m.trainIdx].pt for m in good])

# # Find homography
# homography_matrix, mask = cv.findHomography(points1, points2, cv.RANSAC, 5.0)

# # Register
# h, w= img2.shape[:2]
# img1 = cv.imread('./TestData/normalized.jpg')
# S = np.array([[10, 0, 0], [0, 10, 0], [0, 0 ,1]])
# Sinv = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]])
# homography_matrix_new = np.matmul(S, homography_matrix)
# homography_matrix_new = np.matmul(homography_matrix_new, Sinv)
# registered_image = cv.warpPerspective(img1, homography_matrix_new, (10*w, 10*h))

# cv.imwrite('registered_image-3.jpg',registered_image)
# plt.imshow(registered_image)
# plt.axis('off')
# print(len(good))

# print(homography_matrix)

# np.save('homography_best_method1.npy', homography_matrix_new)


# source_image_path = './TestData/01-HE.jpg'
# source_image = Image.open(source_image_path)
# source_landmarks_path = './TestData/01-HE.csv'
# source_landmarks = read_coordinates_from_csv(source_landmarks_path)

# target_image_path = './TestData/01-CC10.jpg'
# target_image = Image.open(target_image_path)
# target_landmarks_path = './TestData/01-CC10.csv'
# target_landmarks = read_coordinates_from_csv(target_landmarks_path)



# original_width, original_height = target_image.size
# source_width, source_height = source_image.size

# # Calculate padding
# left_padding = (source_width - original_width) // 2
# top_padding = (source_height - original_height) // 2

# # Create a new image with target dimensions and white background
# new_image = Image.new('RGB', (source_width, source_height), (255, 255, 255))

# # Paste the original image onto the new image, centered
# new_image.paste(target_image, (left_padding, top_padding))
# target_image = new_image

# source_image = np.array(source_image)
# target_image = np.array(target_image)
# blended_image = blend_images(source_image, target_image)
# plt.imshow(blended_image)
# plot_landmarks(blended_image, source_landmarks)
# plt.imshow(blended_image)
# plot_landmarks(blended_image, target_landmarks, color='blue')


# # Reshape the landmarks to a format compatible with cv2.perspectiveTransform
# target_landmarks = np.array(target_landmarks).reshape(-1, 1, 2)

# # Transform the coordinates
# transformed_landmarks = cv.perspectiveTransform(target_landmarks, homography_matrix_new)

# # Reshape back to original format
# transformed_landmarks = transformed_landmarks.reshape(-1, 2)
# transformed_landmarks_list = [tuple(np.round(coord,1)) for coord in transformed_landmarks]

# plot_landmarks(blended_image, transformed_landmarks_list, color='green')

# def plot_landmarks_with_overlay(image, source_landmarks, target_landmarks, transformed_landmarks):
#     """
#     Plot landmarks on an image and draw lines between corresponding points.

#     Args:
#     - image: The image array.
#     - source_landmarks: List of (x, y) coordinates for source landmarks.
#     - target_landmarks: List of (x, y) coordinates for target landmarks.
#     - transformed_landmarks: List of (x, y) coordinates for transformed source landmarks.
#     """
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)

#     # Plot lines before registration
#     #for (src, tgt) in zip(source_landmarks, target_landmarks):
#      #   plt.plot([src[0], tgt[0]], [src[1], tgt[1]], 'r--', label='Before Registration' if src == source_landmarks[0] else "")

#     # Plot lines after registration
#     #for (src_trans, tgt) in zip(transformed_landmarks, target_landmarks):
#     #    plt.plot([src_trans[0], tgt[0]], [src_trans[1], tgt[1]], 'b-', label='After Registration' if src_trans == transformed_landmarks[0] else "")

#     # Plot source landmarks
#     x_coords, y_coords = zip(*source_landmarks)
#     plt.scatter(x_coords, y_coords, color='r', label='Source Landmarks')

#     # Plot target landmarks
#     # x_coords, y_coords = zip(*target_landmarks)
#     # plt.scatter(x_coords, y_coords, color='b', label='Target Landmarks')
    
#     # Plot transformed landmarks
#     x_coords, y_coords = zip(*transformed_landmarks)
#     plt.scatter(x_coords, y_coords, color='g', label='Transformed Landmarks')
    

# plot_landmarks_with_overlay(blended_image, source_landmarks, target_landmarks, transformed_landmarks)