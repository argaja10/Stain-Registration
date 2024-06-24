
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def get_mean_and_std(x):
    """
    Compute the mean and standard deviation of each channel in the given image.
    
    Args:
        image (numpy.ndarray): The input image.
    
    Returns:
        tuple: Means and standard deviations of the image channels.
    """
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean,2))
    x_std = np.hstack(np.around(x_std,2)) 
    return x_mean, x_std



def get_normalized_image(input_img_path):
    """
    Compute the stain normalized image using Reinhard transformation.

    Args:
        input_img_path: The input image path.
    
    Returns:
        image (numpy.ndarray): Normalized image.
    """
    template_img = cv2.imread('./TestData/01-HE.jpg')
    template_img = cv2.cvtColor(template_img,cv2.COLOR_BGR2LAB)
    template_mean, template_std = get_mean_and_std(template_img)
    
    input_img = cv2.imread(input_img_path)
    input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2LAB)
    img_mean, img_std = get_mean_and_std(input_img)
    
    height, width, channel = input_img.shape
    for i in range(0,height):
        for j in range(0,width):
            for k in range(0,channel):
                x = input_img[i,j,k]
                x = ((x-img_mean[k])*(template_std[k]/img_std[k]))+template_mean[k]
                x = round(x)
                # boundary check
                x = 0 if x<0 else x
                x = 255 if x>255 else x
                input_img[i,j,k] = x
                
    normalized_img= cv2.cvtColor(input_img,cv2.COLOR_LAB2BGR)
    #plt.imshow(input_img)
    # Get the image name from the path
    img_name = os.path.basename(input_img_path)
    normalized_img_path = './TestData/normalized-' + img_name
    cv2.imwrite(normalized_img_path, normalized_img)
    
    return normalized_img, normalized_img_path
    
    

# Stain normalization of IHC markers

normalized_img, normalized_img_path = get_normalized_image('./TestData/01-CC10.jpg')
plt.imshow(normalized_img)

normalized_img, normalized_img_path = get_normalized_image('./TestData/01-CD31.jpg')
plt.imshow(normalized_img)

normalized_img, normalized_img_path = get_normalized_image('./TestData/01-Ki67.jpg')
plt.imshow(normalized_img)

