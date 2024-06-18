# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 08:45:13 2024

@author: argajashende
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import euclidean
from visualization import plot_landmarks, read_coordinates_from_csv, blend_images




def visualize_overlay_landmarks_with_distances(source_image, target_image, source_landmarks, target_landmarks, transform=None):
    """
    Visualize the overlay of two images with landmarks and show the distances before and after registration.
    
    Args:
    - source_image: The source image array.
    - target_image: The target image array.
    - source_landmarks: List of (x, y) coordinates for the source image landmarks.
    - target_landmarks: List of (x, y) coordinates for the target image landmarks.
    - transform: A function that applies the registration transformation to source landmarks.
    """
    # Blend the images
    blended_image = blend_images(source_image, target_image, alpha=0.5)
    
    # Calculate distances and transformed landmarks
    distances_before = []
    distances_after = []
    transformed_landmarks = []

    for src, tgt in zip(source_landmarks, target_landmarks):
        distances_before.append(euclidean(src, tgt))
        
        if transform:
            src_transformed = transform(np.array(src))
        else:
            src_transformed = np.array(src)
        
        distances_after.append(euclidean(src_transformed, np.array(tgt)))
        transformed_landmarks.append(src_transformed)

    # Plot blended image with landmarks and lines
    plt.figure(figsize=(10, 10))
    plt.imshow(blended_image)

    # Plot original landmarks
    for (src, tgt) in zip(source_landmarks, target_landmarks):
        plt.plot([src[0], tgt[0]], [src[1], tgt[1]], 'r--')
    plot_landmarks(blended_image, source_landmarks, color='r', label='Source Landmarks')

    # Plot target landmarks
    plot_landmarks(blended_image, target_landmarks, color='g', label='Target Landmarks')

    # Plot transformed source landmarks
    for (src_trans, tgt) in zip(transformed_landmarks, target_landmarks):
        plt.plot([src_trans[0], tgt[0]], [src_trans[1], tgt[1]], 'b-')
    plot_landmarks(blended_image, transformed_landmarks, color='b', label='Transformed Source Landmarks')

    plt.title('Overlay of Source and Target Images with Landmarks')
    plt.show()

    # Print distances
    print(f"Distances Before Registration: {distances_before}")
    print(f"Distances After Registration: {distances_after}")

# Example usage:
source_image_path = 'path_to_source_image.png'
target_image_path = 'path_to_target_image.png'

source_image = np.array(Image.open(source_image_path))
target_image = np.array(Image.open(target_image_path))

source_landmarks = [(30, 50), (80, 90), (130, 150)]
target_landmarks = [(32, 52), (78, 88), (128, 148)]

# Define a dummy registration transformation for demonstration.
def dummy_transform(point):
    return point + np.array([2, 2])

visualize_overlay_landmarks_with_distances(source_image, target_image, source_landmarks, target_landmarks, transform=dummy_transform)
