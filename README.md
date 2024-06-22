# stain_registration_v2
Accurate registration of multiple whole slide images (WSI) is vital for several pathology tasks
such as joint analysis of various biomarkers and study of tissue morphology to aid in early diagnosis
of diseases, continuous monitoring and patient specific treatment planning. This project investigates
different methods to register immunohistochemical (IHC) stained images to hematoxylin and eosin
(H&E) stained images and evaluates their performance. The methods are tested on a dataset con-
sisting of one set of stains (a total of four images) of serial sections stained with H&E and three IHC
markers (CC10, CD31, and Ki67), along with their corresponding landmark coordinates.
## Dataset
The data set used to test the different methods, consists of four images of consecutive tissue slices, stained
with H&E and three IHC markers, namely CC10, CD31 and Ki67. Each of the images are provided with
associated landmarks denoting key points which are used to evaluate the performance of the registration
methods. 
<div align="center">
	<img width = "20%" src="stain_registration/TestData/01-HE.jpg">
  <img width = "20%" src="stain_registration/TestData/01-CC10.jpg">
  <img width = "20%" src="stain_registration/TestData/01-CD31.jpg">
  <img width = "20%" src="stain_registration/TestData/01-Ki67.jpg">
</div>

## Method
This project implements a combination of intensity and feature based methods and evaluates its performance on the test data based
on mean error (average euclidean distance between registered and true landmarks), k-Pixel threshold error (percentage of pixels for which the registered landmarks is off the
ground truth by more than k pixels) and average computational time (time to perform the registration including finding the transformation matrix/parameters
and transforming the landmarks).
The registrations methods used in this work are scale invariant feature transform (SIFT), optical flow and ANTs. The images are preprocessed to normalize the stain using the Reinhard color transformation technique, using the H&E stain as the template.

## Structure
Below is a high-level overview of the project structure:

''' css
your-repo-name/
├── images/
│   ├── screenshot1.png
│   ├── screenshot2.png
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
│   ├── analysis.ipynb
├── src/
│   ├── main.py
│   ├── module1.py
│   ├── module2.py
├── tests/
│   ├── test_main.py
│   ├── test_module1.py
├── .gitignore
├── environment.yml
├── LICENSE
├── README.md
'''
