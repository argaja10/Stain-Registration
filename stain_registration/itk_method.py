# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:19:49 2024

@author: argajashende
"""


import ants
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2
from skimage import io, transform
from visualization import plot_landmarks, blend_images, read_coordinates_from_csv, plot_landmarks_with_overlay
import pandas as pd

# Method 3: Using intensity-based antspy library to register images using SyNRA transformation method

def ants_registration(fixed_image_path, moving_image_path, target_landmarks, type_of_transform):
    fixed_image_gray = np.array(ImageOps.grayscale((Image.open(fixed_image_path))))
    moving_image_gray = np.array(ImageOps.grayscale((Image.open(moving_image_path))))

    # Convert grayscale images to ANTs images
    fixed_image = ants.from_numpy(fixed_image_gray)
    moving_image = ants.from_numpy(moving_image_gray)

    # Perform affine registration on grayscale images
    registration_result = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyNRA')

    # Get the warped moving image
    #warped = registration_result['warpedmovout']
    warped = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=registration_result['fwdtransforms'])

    warped = warped.numpy()
    
    target_landmarks = target_landmarks[:,::-1]
    target_landmarks= pd.DataFrame(target_landmarks, columns=['x', 'y'])

    #warped_moving = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=registration_result['fwdtransforms'])
    transformed_points = ants.apply_transforms_to_points(dim=2, points=target_landmarks, transformlist=registration_result['invtransforms'])

    transformed_points = transformed_points.to_numpy()[:,::-1]
    transformed_points = [tuple(r) for r in transformed_points]
    
    # # Extract the transformation matrix
    # transformation_matrix = registration_result['fwdtransforms'][1]
    # # Read the transformation matrix
    # transformation = ants.read_transform(registration_result['fwdtransforms'][1])

    # # Extract the transformation matrix
    # # Convert the transformation to a numpy array (assuming a 2D affine transform)
    # transform_matrix = transformation.parameters

    # # Convert the 1D transform parameters to a 2D affine matrix
    # # Note: This assumes the transformation is 2D affine with 6 parameters
    # # For 3D affine transforms, the number of parameters will be different
    # transform_matrix = np.array([
    #     [transform_matrix[0], transform_matrix[1], transform_matrix[4]],
    #     [transform_matrix[2], transform_matrix[3], transform_matrix[5]],
    #     [0, 0, 1]
    # ])

    # # Convert moving points to homogeneous coordinates (add a column of ones)
    # homogeneous_moving_points = np.hstack([-1*target_landmarks[:,::-1], np.ones((target_landmarks.shape[0], 1))])


    # # Apply the transformation matrix to the points
    # transformed_homogeneous_points = homogeneous_moving_points @ transform_matrix.T

    # # Convert back from homogeneous coordinates
    # transformed_points = transformed_homogeneous_points[:, :2]
    # transformed_points = -1*transformed_points[:,::-1]
    
    return warped, transformed_points
    

# Case 1: Registration of CC10 to H&E stain
fixed_image_path = './TestData/01-HE.jpg'  # H&E stained image
moving_image_path = './TestData/normalized-01-CC10.jpg'  # IHC stained normalized image

fixed_image = np.array(Image.open('./TestData/01-HE.jpg'))
source_landmarks = read_coordinates_from_csv('./TestData/01-HE.csv')

target_landmarks = read_coordinates_from_csv('./TestData/01-CC10.csv')
target_landmarks = np.array(target_landmarks)

warped_moving, transformed_points = ants_registration(fixed_image_path, moving_image_path, target_landmarks, type_of_transform='SyNRA')

plt.imshow(warped_moving, cmap='gray')


plot_landmarks_with_overlay(fixed_image, source_landmarks, target_landmarks, transformed_points, label='Before Registration')

plot_landmarks_with_overlay(fixed_image, source_landmarks, target_landmarks, transformed_points, label='After Registration')


# Case 2: Registration of CD31 to H&E stain
fixed_image_path = './TestData/01-HE.jpg'  # H&E stained image
moving_image_path = './TestData/normalized-01-CD31.jpg'  # IHC stained normalized image

fixed_image = np.array(Image.open('./TestData/01-HE.jpg'))
source_landmarks = read_coordinates_from_csv('./TestData/01-HE.csv')

target_landmarks = read_coordinates_from_csv('./TestData/01-CD31.csv')
target_landmarks = np.array(target_landmarks)

warped_moving, transformed_points = ants_registration(fixed_image_path, moving_image_path, target_landmarks, type_of_transform='SyNRA')

plt.imshow(warped_moving, cmap='gray')


plot_landmarks_with_overlay(fixed_image, source_landmarks, target_landmarks, transformed_points, label='Before Registration')

plot_landmarks_with_overlay(fixed_image, source_landmarks, target_landmarks, transformed_points, label='After Registration')


# Case 3: Registration of Ki67 to H&E stain
fixed_image_path = './TestData/01-HE.jpg'  # H&E stained image
moving_image_path = './TestData/normalized-01-Ki67.jpg'  # IHC stained normalized image

fixed_image = np.array(Image.open('./TestData/01-HE.jpg'))
source_landmarks = read_coordinates_from_csv('./TestData/01-HE.csv')

target_landmarks = read_coordinates_from_csv('./TestData/01-Ki67.csv')
target_landmarks = np.array(target_landmarks)

warped_moving, transformed_points = ants_registration(fixed_image_path, moving_image_path, target_landmarks, type_of_transform='SyNRA')

plt.imshow(warped_moving, cmap='gray')



plot_landmarks_with_overlay(fixed_image, source_landmarks, target_landmarks, transformed_points, label='Before Registration')

plot_landmarks_with_overlay(fixed_image, source_landmarks, target_landmarks, transformed_points, label='After Registration')











# # Load the two RGB images
# fixed_image_path = './TestData/01-HE.jpg'  # H&E stained image
# moving_image_path = './TestData/normalized-01-CC10.jpg'  # IHC stained normalized image

# fixed_image_gray = np.array(ImageOps.grayscale((Image.open(fixed_image_path))))
# moving_image_gray = np.array(ImageOps.grayscale((Image.open(moving_image_path))))

# # Convert grayscale images to ANTs images
# fixed_image = ants.from_numpy(fixed_image_gray)
# moving_image = ants.from_numpy(moving_image_gray)

# # Perform affine registration on grayscale images
# registration_result = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyNRA')

# # Extract the transformation matrix
# transformation_matrix = registration_result['fwdtransforms'][1]
# # Read the transformation matrix
# transformation = ants.read_transform(registration_result['fwdtransforms'][1])

# # Extract the transformation matrix
# # Convert the transformation to a numpy array (assuming a 2D affine transform)
# transform_matrix = transformation.parameters


# # Convert the 1D transform parameters to a 2D affine matrix
# # Note: This assumes the transformation is 2D affine with 6 parameters
# # For 3D affine transforms, the number of parameters will be different
# transform_matrix = np.array([
#     [transform_matrix[0], transform_matrix[1], transform_matrix[4]],
#     [transform_matrix[2], transform_matrix[3], transform_matrix[5]],
#     [0, 0, 1]
# ])

# warped_moving = registration_result['warpedmovout']
# warped_moving = warped_moving.numpy()
# plt.imshow(warped_moving, cmap='gray')
# plt.imshow(fixed_image_gray, cmap='gray')
# plt.imshow(moving_image_gray, cmap='gray')

# source_image_path = './TestData/01-HE.jpg'
# source_image = Image.open(source_image_path)
# source_landmarks_path = './TestData/01-HE.csv'
# source_landmarks = read_coordinates_from_csv(source_landmarks_path)

# target_image_path = './TestData/01-CC10.jpg'
# target_image = Image.open(target_image_path)
# target_landmarks_path = './TestData/01-CC10.csv'
# target_landmarks = read_coordinates_from_csv(target_landmarks_path)
# target_landmarks = np.array(target_landmarks)


# # Convert moving points to homogeneous coordinates (add a column of ones)
# #homogeneous_moving_points = np.hstack([-1*target_landmarks[:,::-1], np.ones((target_landmarks.shape[0], 1))])

# homogeneous_moving_points = np.hstack([target_landmarks, np.ones((target_landmarks.shape[0], 1))])
# # Apply the transformation matrix to the points
# transformed_homogeneous_points = homogeneous_moving_points @ np.linalg.inv(transform_matrix).T

# # Convert back from homogeneous coordinates
# transformed_points = transformed_homogeneous_points[:, :2]
# #transformed_points = -1*transformed_points[:,::-1]


# # Output the transformed points
# print("Transformed Points:", transformed_points)

# fixed_image = np.array(Image.open(fixed_image_path))
# plot_landmarks(fixed_image, source_landmarks)

# plot_landmarks(fixed_image, target_landmarks, color='blue')

# plot_landmarks(fixed_image, transformed_points, color='green')

# plot_landmarks(warped_moving, transformed_points, color='green')


# plot_landmarks_with_overlay(fixed_image, source_landmarks, target_landmarks, transformed_points, label='After Registration')

# plot_landmarks_with_overlay(fixed_image, source_landmarks, target_landmarks, transformed_points, label='Before Registration')

# moving_points_df = pd.DataFrame(target_landmarks, columns=['x', 'y'])
# moving_points_df['z'] = 0  # Add z dimension for 2D points

# # Apply the transformation to the points using the forward transformation
# transformed_points_df = ants.apply_transforms_to_points(
#     2,
#     points=moving_points_df,
#     transformlist=registration_result['fwdtransforms']
# )

# # Convert the transformed points DataFrame back to a numpy array and drop the z column
# transformed_points = transformed_points_df[['x', 'y']].to_numpy()

# plt.imshow(np.array(target_image))

# target_landmarks= pd.DataFrame(target_landmarks, columns=['x', 'y'])
# target_landmarks.columns = ['y', 'x']

# target_landmarks = target_landmarks[:,::-1]
# target_landmarks= pd.DataFrame(target_landmarks, columns=['x', 'y'])

# warped_moving = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=registration_result['fwdtransforms'])
# warped_points = ants.apply_transforms_to_points(dim=2, points=target_landmarks, transformlist=registration_result['invtransforms'])

# plt.imshow(warped_moving.numpy(), cmap='gray')
# warped_points.columns = ['y', 'x']

# transformed_points = warped_points

# transformed_points = transformed_points.to_numpy()[:,::-1]
# transformed_points = [tuple(r) for r in transformed_points]