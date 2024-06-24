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

```
.
├── stain_registration
│   ├── Results
│   │   └── HE_CC10.png
│   │   └── HE_CD31.png
│   │   └── HE_Ki67.png
│   │   └── SIFT_CC10.png
│   │   └── SIFT_CD31.png
│   │   └── SIFT_Ki67.png
│   │   └── antspy_CC10.png
│   │   └── antspy_CD31.png
│   │   └── antspy_Ki67.png
│   │   └── antspy_registered_CC10.png
│   │   └── antspy_registered_CD31.png
│   │   └── antspy_registered_Ki67.png
│   │   └── optical_flow_CC10.png
│   │   └── optical_flow_CD31.png
│   │   └── optical_flow_Ki67.png
│   │   └── optical_flow_registered_CC10.png
│   │   └── optical_flow_registered_CD31.png
│   │   └── optical_flow_registered_Ki67.png
│   ├── TestData
│   │   └── 01-CC10.csv
│   │   └── 01-CC10.jpg
│   │   └── 01-CD31.csv
│   │   └── 01-CD31.jpg
│   │   └── 01-HE.csv
│   │   └── 01-HE.jpg
│   │   └── 01-Ki67.csv
│   │   └── 01-Ki67.jpg
│   │   └── normalized-01-CC10.jpg
│   │   └── normalized-01-CD31.jpg
│   │   └── normalized-01-Ki67.jpg
│   └── evaluation.py
│   └── ants_method.py
│   └── normalize.py
│   └── optical_flow.py
│   └── template_matching_SIFT1.py
│   └── visualization.py
└── LICENSE
└── README.md
└── ants_env.yml
```

| No | File Name | Details 
|----|------------|-------|
| 1  | index | Entry point

## Installation
### Prerequisites

Make sure you have the following installed:

Anaconda (Recommended)

Git

### Setup

Clone the repository:
```
git clone https://github.com/argaja10/stain_registration_v2.git
cd stain_registration_v2
```

Create and activate the conda environment:
```
conda env create -f ants_env.yml
conda activate antspy
```

Verify the installation:

Ensure all dependencies are installed and the environment is set up correctly:
```
conda list
```
