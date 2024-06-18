# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 01:08:29 2024

@author: argajashende
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 10
img1 = cv.imread('./TestData/normalized.jpg', cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread('./TestData/01-HE.jpg', cv.IMREAD_GRAYSCALE) # trainImage


img1 = cv.medianBlur(cv.resize(img1, (0, 0), fx = 0.1, fy = 0.1),5)
img2 = cv.medianBlur(cv.resize(img2, (0, 0), fx = 0.1, fy = 0.1),5)

# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 15)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
 if m.distance < 0.8*n.distance:
     good.append(m)

if len(good)>MIN_MATCH_COUNT:
 src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
 dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
 M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
 matchesMask = mask.ravel().tolist()
 h,w = img1.shape
 pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
 dst = cv.perspectiveTransform(pts,M)
 img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
 print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
 matchesMask = None
 
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
 singlePointColor = None,
 matchesMask = matchesMask, # draw only inliers
 flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.figure(figsize=(10,20), dpi=600)
plt.imshow(img3, 'gray'),plt.show()


# Extract location of matches
points1 = np.float32([kp1[m.queryIdx].pt for m in good])
points2 = np.float32([kp2[m.trainIdx].pt for m in good])

# Find homography
homography_matrix, mask = cv.findHomography(points1, points2, cv.RANSAC, 5.0)

# Register
h, w= img2.shape[:2]
img1 = cv.imread('./TestData/normalized.jpg')
S = np.array([[10, 0, 0], [0, 10, 0], [0, 0 ,1]])
Sinv = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]])
homography_matrix_new = np.matmul(S, homography_matrix)
homography_matrix_new = np.matmul(homography_matrix_new, Sinv)
registered_image = cv.warpPerspective(img1, homography_matrix_new, (10*w, 10*h))

cv.imwrite('registered_image-3.jpg',registered_image)
plt.imshow(registered_image)
plt.axis('off')
print(len(good))

print(homography_matrix)