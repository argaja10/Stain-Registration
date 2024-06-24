# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:19:49 2024

@author: argajashende
"""

import math, random
import ants
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2
from skimage import io, transform
from visualization import plot_landmarks, blend_images, read_coordinates_from_csv, plot_landmarks_with_overlay
import pandas as pd
import random
import os
from evaluation import euclidean_distance_metric, k_pixel_threshold, relative_TRE, robustness
import argparse

os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='1'
# os.environ['ANTS_RANDOM_SEED']='42'
# Method 3: Using intensity-based antspy library to register images using SyNRA transformation method

def ants_registration(fixed_image_path, moving_image_path, target_landmarks, type_of_transform):
    fixed_image_gray = np.array(ImageOps.grayscale((Image.open(fixed_image_path))))
    moving_image_gray = np.array(ImageOps.grayscale((Image.open(moving_image_path))))

    # Convert grayscale images to ANTs images
    fixed_image = ants.from_numpy(fixed_image_gray)
    moving_image = ants.from_numpy(moving_image_gray)
    
    # set seed
    np.random.seed(42)
    random.seed(42)
    
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
    
    return warped, transformed_points
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, required=True, help="Provide file path for source (H&E) image")
    parser.add_argument('--normalized_target_path', type=str, required=True, help="Provide file path for normalized target (IHC) image")
    parser.add_argument('--source_landmarks_path', type=str, required=True, help="Provide file path for source (H&E) landmarks")
    parser.add_argument('--target_landmarks_path', type=str, required=True, help="Provide file path for target (IHC) landmarks")
    
    args = parser.parse_args()
    # Case 1: Registration of CC10 to H&E stain
    #fixed_image_path = './TestData/01-HE.jpg'  # H&E stained image
    #moving_image_path = './TestData/normalized-01-CC10.jpg'  # IHC stained normalized image
    
    fixed_image = np.array(Image.open(args.source_path))
    source_landmarks = read_coordinates_from_csv(args.source_landmarks_path)
    
    target_landmarks = read_coordinates_from_csv(args.target_landmarks_path)
    target_landmarks = np.array(target_landmarks)
    
    warped_moving, transformed_points = ants_registration(args.source_path, args.normalized_target_path, target_landmarks, type_of_transform='SyNRA')
    
    plt.imshow(warped_moving, cmap='gray')
    plt.show()
    
    plot_landmarks_with_overlay(fixed_image, source_landmarks, target_landmarks, transformed_points, label='Before Registration')
    plt.show()
    
    plot_landmarks_with_overlay(fixed_image, source_landmarks, target_landmarks, transformed_points, label='After Registration')
    plt.show()
    
    # Evaluation:
    distances, sum_distances, avg_distances = euclidean_distance_metric(source_landmarks, transformed_points)
    print(f"Avg distance between source and transformed images: {avg_distances}")
    rTRE = relative_TRE(distances, fixed_image)
    kpte = k_pixel_threshold(distances, 50)
    print(f"K-Pixel threshold value: {kpte}")
    robust = robustness(source_landmarks, target_landmarks, transformed_points)
    print(f"Robustness: {robust}")
    
    


if __name__=="__main__":
    main()
    







