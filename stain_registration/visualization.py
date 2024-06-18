# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:57:57 2024

@author: argajashende
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
#from template_matching_SIFT1 import homography_matrix_new
import cv2

def plot_landmarks(image, landmarks, color='r', label='Landmark'):
    """
    Plot landmarks on an image.
    
    Args:
    - image: The image array.
    - landmarks: List of (x, y) coordinates for landmarks.
    - color: Color of the landmarks.
    - label: Label for the landmarks.
    """
    plt.imshow(image)
    x_coords, y_coords = zip(*landmarks)
    plt.scatter(x_coords, y_coords, color=color, label=label, s=7)
    for i, (x, y) in enumerate(landmarks):
        plt.text(x, y, str(i), color='black', fontsize=6)
    #plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()
    

def read_coordinates_from_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    # Extract the X and Y columns and convert them to a list of tuples
    coordinates = list(zip(df['X'], df['Y']))
    return coordinates

def blend_images(image1, image2, alpha=0.5):
    """
    Blend two images with a given alpha transparency.
    
    Args:
    - image1: The first image array.
    - image2: The second image array.
    - alpha: The transparency factor for image1.
    
    Returns:
    - blended_image: The blended image array.
    """
    image1 = Image.fromarray(image1)
    image2 = Image.fromarray(image2)
    
    blended_image = Image.blend(image1, image2, alpha)
    return np.array(blended_image)

def overlay_images(fixed_image, registered_image, alpha=0.5):
    """
    Overlays a fixed image and a registered image with a given alpha value and displays the result.

    Parameters:
    fixed_image_path: Fixed image.
    registered_image_path: Registered image.
    alpha (float): Transparency factor for the overlay. Default is 0.5.

    Returns:
    overlay : Overlayed image.
    """
   
    # Ensure the images are the same size
    if fixed_image.shape != registered_image.shape:
        print("The images are not of the same size. Resizing the registered image to match the fixed image.")
        registered_image = cv2.resize(registered_image, (fixed_image.shape[1], fixed_image.shape[0]))

    # Blend the images with the specified alpha value
    overlay = cv2.addWeighted(fixed_image, alpha, registered_image, 1 - alpha, 0)

    # Display the overlay
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Overlay of Fixed and Registered Images')
    plt.axis('off')
    plt.show()
    
    return overlay

def plot_landmarks_with_overlay(image, source_landmarks, target_landmarks, transformed_landmarks, label='Before Registration'):
    """
    Plot landmarks on an image and draw lines between corresponding points.

    Args:
    - image: The image array.
    - source_landmarks: List of (x, y) coordinates for source landmarks.
    - target_landmarks: List of (x, y) coordinates for target landmarks.
    - transformed_landmarks: List of (x, y) coordinates for transformed source landmarks.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    if label == 'Before Registration':
        # Plot lines before registration
        for (src, tgt) in zip(source_landmarks, target_landmarks):
            plt.plot([src[0], tgt[0]], [src[1], tgt[1]], 'b--', label='Before Registration')
        # Plot source landmarks
        x_coords, y_coords = zip(*source_landmarks)
        plt.scatter(x_coords, y_coords, color='r', label='Source Landmarks')

        # Plot target landmarks
        x_coords, y_coords = zip(*target_landmarks)
        plt.scatter(x_coords, y_coords, color='b', label='Target Landmarks')
        
    elif label == 'After Registration':
        # Plot lines after registration
        for (src_trans, src) in zip(transformed_landmarks, source_landmarks):
            plt.plot([src_trans[0], src[0]], [src_trans[1], src[1]], 'g--', label='After Registration' )
        # Plot source landmarks
        x_coords, y_coords = zip(*source_landmarks)
        plt.scatter(x_coords, y_coords, color='r', label='Source Landmarks')
        
        # Plot transformed landmarks
        x_coords, y_coords = zip(*transformed_landmarks)
        plt.scatter(x_coords, y_coords, color='g', label='Transformed Landmarks')
        
    




























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
# transformed_landmarks = cv2.perspectiveTransform(target_landmarks, homography_matrix_new)

# # Reshape back to original format
# transformed_landmarks = transformed_landmarks.reshape(-1, 2)
# transformed_landmarks_list = [tuple(np.round(coord,1)) for coord in transformed_landmarks]

# plot_landmarks(blended_image, transformed_landmarks_list, color='green')