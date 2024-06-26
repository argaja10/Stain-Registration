# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:54:07 2024

@author: argajashende
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage.registration import optical_flow_tvl1
from skimage.color import rgb2gray
import cv2 as cv
from visualization import plot_landmarks, blend_images, read_coordinates_from_csv, plot_landmarks_with_overlay
from PIL import Image
from evaluation import euclidean_distance_metric, k_pixel_threshold, relative_TRE, robustness
import os, argparse

def resize_images(source, target):
    # Resize target image to the size of source image
    target_resized = transform.resize(target, source.shape, anti_aliasing=True)
    return target_resized

def compute_optical_flow(source, target):
    # Compute the optical flow between source and target images
    v, u = optical_flow_tvl1(source, target,attachment=5, tightness=0.1, num_warp=30, num_iter=200, tol=1e-4, prefilter=True)
    return v, u

def warp_image(target, v, u):
    # Warp the target image using the estimated optical flow
    nr, nc = target.shape[0], target.shape[1]
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    warp_coords = np.array([row_coords + v, col_coords + u])
    target_warped = transform.warp(target, warp_coords, mode='edge')
    return target_warped

def register_images(source, target):
    
    source_dwn = cv.medianBlur(cv.resize(source, (0, 0), fx = 0.1, fy = 0.1), 5)
    target_dwn = cv.medianBlur(cv.resize(target, (0, 0), fx = 0.1, fy = 0.1),5)
    target_resized = resize_images(source_dwn, target_dwn)
    v, u = compute_optical_flow(source_dwn, target_resized)
    
    h_old, w_old = u.shape[:2]
    h_new, w_new = target.shape[:2]
    from scipy.interpolate import interp2d
    #print(u.shape, v.shape)
    # Interpolating and rescaling u and v
    f_u = interp2d(np.linspace(0,w_old-1,w_old),np.linspace(0,h_old-1,h_old),10*u,kind='linear')
    f_v = interp2d(np.linspace(0,w_old-1,w_old),np.linspace(0,h_old-1,h_old),10*v,kind='linear')
    
    u_rescaled = f_u(np.linspace(0,w_old-1,w_new),np.linspace(0,h_old-1,h_new))
    v_rescaled = f_v(np.linspace(0,w_old-1,w_new),np.linspace(0,h_old-1,h_new))
    
    target_warped = warp_image(target, v_rescaled, u_rescaled)
    
    return target_warped, v_rescaled, u_rescaled

def transform_landmarks(landmarks, v, u):
    # Transform the landmark coordinates using the optical flow vectors
    transformed_landmarks = landmarks.copy()
    for i, (x, y) in enumerate(landmarks):
        dx, dy = u[int(y), int(x)], v[int(y), int(x)]
        transformed_landmarks[i] = [x - dx, y - dy]
    return transformed_landmarks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, required=True, help="Provide file path for source (H&E) image")
    parser.add_argument('--normalized_target_path', type=str, required=True, help="Provide file path for normalized target (IHC) image")
    parser.add_argument('--source_landmarks_path', type=str, required=True, help="Provide file path for source (H&E) landmarks")
    parser.add_argument('--target_landmarks_path', type=str, required=True, help="Provide file path for target (IHC) landmarks")
    
    args = parser.parse_args()
    
    # Case 1: Registration of CC10 to H&E stain
    # Load source and target images
    source = cv.imread(args.source_path, cv.IMREAD_GRAYSCALE)
    target = cv.imread(args.normalized_target_path, cv.IMREAD_GRAYSCALE)
    
    # Register the images
    registered_image, v_rescaled, u_rescaled = register_images(source, target)
    plt.imshow(registered_image, cmap='gray')
    plt.show()
    
    source_image = np.array(Image.open(args.source_path))
    #source_landmarks_path = './TestData/01-HE.csv'
    source_landmarks = read_coordinates_from_csv(args.source_landmarks_path)
    
    #target_landmarks_path = './TestData/01-CC10.csv'
    target_landmarks = read_coordinates_from_csv(args.target_landmarks_path)
    
    transformed_landmarks = transform_landmarks(target_landmarks, v_rescaled, u_rescaled) 
    plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks, label='Before Registration')
    plt.show()
    plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks, label='After Registration')
    plt.show()

    # Evaluation:
    distances, sum_distances, avg_distances = euclidean_distance_metric(source_landmarks, transformed_landmarks)
    print(f"Avg distance between source and transformed images: {avg_distances}")
    rTRE = relative_TRE(distances, source_image)
    kpte = k_pixel_threshold(distances, 50)
    print(f"K-Pixel threshold value: {kpte}")
    robust = robustness(source_landmarks, target_landmarks, transformed_landmarks)
    print(f"Robustness: {robust}")


if __name__ == "__main__":
    main()









