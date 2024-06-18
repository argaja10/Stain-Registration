# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:03:48 2024

@author: argajashende
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def detect_orb_features(source, template, max_features):
    # Convert images to grayscale
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Dectect ORB feature and compute descriptors
    orb = cv2.ORB_create(max_features)
    keypts1, dsc1 = orb.detectAndCompute(source_gray, None)
    keypts2, dsc2 = orb.detectAndCompute(template_gray, None)
    
    return keypts1, dsc1, keypts2, dsc2

def draw_matches(source, template, keypts1, dsc1, keypts2, dsc2):
    # Hamming distance to match the descriptor
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.knnMatch(dsc1, dsc2, k=2)
    #matches = sorted(matches, key = lambda x:x.distance)
    # Sort matches by score
    
    #matches.sort(key=lambda x: x.distance, reverse=False)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
  # Draw matches
    matches_image = cv2.drawMatches(source, keypts1, template, keypts2, good,None,flags=cv2.DrawMatchesFlags_DEFAULT | cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(10,20), dpi=600)
    plt.imshow(matches_image)
    
    return matches
    
def register_images(source, template):
    max_features = 5000
    # Compute keypoints and descriptors
    keypts1, dsc1, keypts2, dsc2 = detect_orb_features(source, template, max_features)
    
    # Compute matches
    matches = draw_matches(source, template, keypts1, dsc1, keypts2, dsc2)
    
    # Extract location of matches
    points1 = np.float32([keypts1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypts2[m.trainIdx].pt for m in matches])

    # Find homography
    homography_matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

    # Register
    h, w= template.shape[:2]
    registered_image = cv2.warpPerspective(source, homography_matrix, (w, h))
    plt.imshow(registered_image)
    #print(registered_image.shape)
    #return matches

source_image_path = './TestData/median_CC10.jpg'
source = np.array(Image.open(source_image_path))

template_image_path = './TestData/median_HE.jpg'
template = np.array(Image.open(template_image_path))


register_images(source, template)

