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
	<img width = "15%" src="stain_registration/TestData/01-HE.jpg">
  <img width = "15%" src="stain_registration/TestData/01-CC10.jpg">
  <img width = "15%" src="stain_registration/TestData/01-CD31.jpg">
  <img width = "15%" src="stain_registration/TestData/01-Ki67.jpg">
</div>

