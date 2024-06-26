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
from skimage.metrics import hausdorff_distance



def euclidean_distance_metric(landmarks1, landmarks2):
    """
    Calculate the Euclidean distance between corresponding landmarks in two sets.

    Parameters:
    landmarks1, landmarks2: Arrays of shape (N, 2) representing N landmarks with (x, y) coordinates.

    Returns:
    distances: Array of Euclidean distances between corresponding landmarks.
    """
    assert len(landmarks1) == len(landmarks2), "The number of landmarks in both sets must be equal."
    landmarks1 = np.array(landmarks1)
    landmarks2 = np.array(landmarks2)
    # Calculate Euclidean distances
    distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)
    sum_distances = np.sum(distances)
    avg_distances = np.mean(distances)

    return distances, sum_distances, avg_distances



def k_pixel_threshold(distances, k):
    """
    Calculates the percentage of pixels for which the predicted disparity is off the ground truth by more than k pixels.

    Parameters:
    distances : List of euclidean dstances between the landmarks.
    k : Pixel threshold.

    Returns:
    percentage : Percentage of landmarks that are offset by more than k-pixels.
    """
    count = len([d for d in distances if d > k])
    #print(count)
    percentage = (count/len(distances))*100
    
    return percentage
    

def relative_TRE(distances, source_image):
    """
    Calculate the length of the diagonal of an image.

    Parameters:
    image: Source image to calculate the diagonal length.

    Returns:
    rdistances: List of relative target registrtaion error for each landmark.
    """
    # Get the dimensions of the image (height, width, number of color channels)
    height, width = source_image.shape[:2]
    
    # Calculate the diagonal length 
    diagonal = np.sqrt(width**2 + height**2)
    
    rdistances = np.mean(distances/diagonal)

    return rdistances


def robustness(landmarks1, landmarks2, landmarks3):
    """
    Parameters
    ----------
    landmarks1 : Landmarks of the source image.
    landmarks2 : Landmarks of the target image.
    landmarks3 : Landmarks of the registered image.

    Returns
    -------
    robustness : Percentage describing how many landmarks improved its TRE after registration.
    """
    dist_source_target, _, _ = euclidean_distance_metric(landmarks1, landmarks2)
    dist_source_transformed, _, _ = euclidean_distance_metric(landmarks1, landmarks3)
    #print(dist_source_target)
    #print(dist_source_transformed)
    count = 0
    total_landmarks =  len(dist_source_target)
    for i in range(total_landmarks):
        if (2*dist_source_transformed[i] < dist_source_target[i]):
            count =  count + 1
    robustness = count/total_landmarks
    
    return robustness
  
    















