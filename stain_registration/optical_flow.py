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
    print(u.shape, v.shape)
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


# Case 1: Registration of CC10 to H&E stain
# Load source and target images
source = cv.imread('./TestData/01-HE.jpg', cv.IMREAD_GRAYSCALE)
target = cv.imread('./TestData/normalized-01-CC10.jpg', cv.IMREAD_GRAYSCALE)

# Register the images
registered_image, v_rescaled, u_rescaled = register_images(source, target)
plt.imshow(registered_image, cmap='gray')

source_image = np.array(Image.open('./TestData/01-HE.jpg'))
source_landmarks_path = './TestData/01-HE.csv'
source_landmarks = read_coordinates_from_csv(source_landmarks_path)

target_landmarks_path = './TestData/01-CC10.csv'
target_landmarks = read_coordinates_from_csv(target_landmarks_path)

transformed_landmarks = transform_landmarks(target_landmarks, v_rescaled, u_rescaled) 
plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks, label='Before Registration')
plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks, label='After Registration')


# Case 2: Registration of CD31 to H&E stain
# Load source and target images
source = cv.imread('./TestData/01-HE.jpg', cv.IMREAD_GRAYSCALE)
target = cv.imread('./TestData/normalized-01-CD31.jpg', cv.IMREAD_GRAYSCALE)

# Register the images
registered_image, v_rescaled, u_rescaled = register_images(source, target)
plt.imshow(registered_image, cmap='gray')

source_image = np.array(Image.open('./TestData/01-HE.jpg'))
source_landmarks_path = './TestData/01-HE.csv'
source_landmarks = read_coordinates_from_csv(source_landmarks_path)

target_landmarks_path = './TestData/01-CD31.csv'
target_landmarks = read_coordinates_from_csv(target_landmarks_path)

transformed_landmarks = transform_landmarks(target_landmarks, v_rescaled, u_rescaled) 
plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks, label='Before Registration')
plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks, label='After Registration')


# Case 3: Registration of Ki67 to H&E stain
# Load source and target images
source = cv.imread('./TestData/01-HE.jpg', cv.IMREAD_GRAYSCALE)
target = cv.imread('./TestData/normalized-01-Ki67.jpg', cv.IMREAD_GRAYSCALE)

# Register the images
registered_image, v_rescaled, u_rescaled = register_images(source, target)
plt.imshow(registered_image, cmap='gray')

source_image = np.array(Image.open('./TestData/01-HE.jpg'))
source_landmarks_path = './TestData/01-HE.csv'
source_landmarks = read_coordinates_from_csv(source_landmarks_path)

target_landmarks_path = './TestData/01-Ki67.csv'
target_landmarks = read_coordinates_from_csv(target_landmarks_path)

transformed_landmarks = transform_landmarks(target_landmarks, v_rescaled, u_rescaled) 
plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks, label='Before Registration')
plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks, label='After Registration')













# # Load source and target images
# source = cv.imread('./TestData/01-HE.jpg', cv.IMREAD_GRAYSCALE)
# target = cv.imread('./TestData/normalized.jpg', cv.IMREAD_GRAYSCALE)


# # Register the images
# registered_image, v_rescaled, u_rescaled = register_images(source, target)

# plt.imshow(registered_image, cmap='gray')

# plt.imshow(source)

# plt.imshow(target)

# def transform_landmarks(landmarks, v, u):
#     # Transform the landmark coordinates using the optical flow vectors
#     transformed_landmarks = landmarks.copy()
#     for i, (x, y) in enumerate(landmarks):
#         dx, dy = u[int(y), int(x)], v[int(y), int(x)]
#         transformed_landmarks[i] = [x - dx, y - dy]
#     return transformed_landmarks




# source_image_path = './TestData/01-HE.jpg'
# source_image = Image.open(source_image_path)
# source_landmarks_path = './TestData/01-HE.csv'
# source_landmarks = read_coordinates_from_csv(source_landmarks_path)

# target_image_path = './TestData/01-CC10.jpg'
# target_image = Image.open(target_image_path)
# target_landmarks_path = './TestData/01-CC10.csv'
# target_landmarks = read_coordinates_from_csv(target_landmarks_path)

# source_image = np.array(source_image)
# target_image = np.array(target_image)

# transformed_landmarks_m2 = transform_landmarks(target_landmarks, v_rescaled, u_rescaled) 

# plot_landmarks(source_image, source_landmarks)

# plot_landmarks(source_image, target_landmarks, color='blue')

# plot_landmarks(source_image, transformed_landmarks_m2, color='green')

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
    

# plot_landmarks_with_overlay(source_image, source_landmarks, target_landmarks, transformed_landmarks_m2)

