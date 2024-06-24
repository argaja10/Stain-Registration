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
from evaluation import euclidean_distance_metric, k_pixel_threshold, relative_TRE, robustness
import os, argparse

def SIFT_registration(source_image_path, normalized_target_image_path, target_image_path, target_landmarks, flag=0):
    # Preprocessing images
    # Read the images in grayscle, downscale and apply median blur
    source_image = cv.imread(source_image_path, cv.IMREAD_GRAYSCALE) 
    target_image = cv.imread(normalized_target_image_path, cv.IMREAD_GRAYSCALE)
    
    if (flag==0):
        
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
        
        # Transform the coordinates
        transformed_landmarks = cv.perspectiveTransform(np.array(target_landmarks).reshape(-1, 1, 2), homography_matrix_rescaled)
    
        # Reshape back to original format
        transformed_landmarks = transformed_landmarks.reshape(-1, 2)
        transformed_landmarks_list = [tuple(np.round(coord,1)) for coord in transformed_landmarks]
    
    elif (flag==1):
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

        # Transform the coordinates
        transformed_landmarks = cv.perspectiveTransform(np.array(target_landmarks).reshape(-1, 1, 2), homography_matrix_rescaled)

        # Reshape back to original format
        transformed_landmarks = transformed_landmarks.reshape(-1, 2)
        transformed_landmarks_list = [tuple(np.round(coord,1)) for coord in transformed_landmarks]
        
    else:
        if flag not in [0, 1]:
            raise ValueError("Flag must be either 0 or 1.")
    
    return registered_image, transformed_landmarks_list




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, required=True, help="Provide file path for source (H&E) image")
    parser.add_argument('--normalized_target_path', type=str, required=True, help="Provide file path for normalized target (IHC) image")
    parser.add_argument('--target_path', type=str, required=True, help="Provide file path for normalized target (IHC) image")
    parser.add_argument('--source_landmarks_path', type=str, required=True, help="Provide file path for source (H&E) landmarks")
    parser.add_argument('--target_landmarks_path', type=str, required=True, help="Provide file path for target (IHC) landmarks")
    parser.add_argument('--flag', type=int, required=True, help="Set to 1 if target image is of Ki67 marker")
    args = parser.parse_args()
    
    # Case 1: Registration of CC10 to H&E stain
    #source_image_path = './TestData/01-HE.jpg'
    source_image = np.array(Image.open(args.source_path))
    #source_landmarks_path = './TestData/01-HE.csv'
    source_landmarks = read_coordinates_from_csv(args.source_landmarks_path)
    
    #target_image_path = './TestData/01-CC10.jpg'
    #target_landmarks_path = './TestData/01-CC10.csv'
    target_landmarks = read_coordinates_from_csv(args.target_landmarks_path)
    
    #normalized_target_image_path = './TestData/normalized-01-CC10.jpg'
    
    registered_image, transformed_landmarks_list = SIFT_registration(args.source_path, args.normalized_target_path, args.target_path, target_landmarks, args.flag)
    
    plt.imshow(registered_image)
    plt.show()
    
    plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks_list, label='Before Registration')
    plt.show()
    
    plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks_list, label='After Registration')
    plt.show()
    
    # Evaluation:
    distances, sum_distances, avg_distances = euclidean_distance_metric(source_landmarks, transformed_landmarks_list)
    print(f"Avg distance between source and transformed images: {avg_distances}")
    rTRE = relative_TRE(distances, source_image)
    kpte = k_pixel_threshold(distances, 50)
    print(f"K-Pixel threshold value: {kpte}")
    robust = robustness(source_landmarks, target_landmarks, transformed_landmarks_list)
    print(f"Robustness: {robust}")
    
    
if __name__ == "__main__":
    main()    







