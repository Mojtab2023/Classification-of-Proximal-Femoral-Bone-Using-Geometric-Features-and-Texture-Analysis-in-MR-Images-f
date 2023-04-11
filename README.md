# Classification-of-Proximal-Femoral-Bone-Using-Geometric-Features-and-Texture-Analysis-in-MR-Images-f
The aim of this study was to use geometric features and texture analysis in MR Images in combination with genetic selection algorithm (GSA) and machine learning to determine bone quality and discriminate between healthy and unhealthy bones.
How to use the code scripts for classification and GA_based feature selection?

1.	Doing classification using all the extracted features:

Run the code Classify_By_All.m

Be sure that you revise the variable “Address” at the beginning the code. It should point to the directory that contain the feature vector “FV.xlsx” and targets “Targets.xlsx”

The output of this script is the 10*5 matrix “Acc”. While its columns refer to the 5 employed classifier (SVM_linear, SVM_rbf, FUZZY C-means, Decision Tree, and Logistic Regression), its rows show the experiment index. All the experiments performed 10 times and the result will show using a boxplot.

2.	Doing classification using the selected 56 features by the GA_SVM_rbf:

Run the code script Classify_By_Sel_features.m

Be sure that you revise the variable “Address” at the beginning the code. It should point to the directory that contain the feature vector “FV.xlsx” and targets “Targets.xlsx”

The variable Sel_features point to the features that have been optimized by the GA and reached the best accuracy of 89% using SVM_rbf.

The output description is similar to section one.

3.	Genetic Algorithm based feature selection coupled with SVM (with various types of kernel functions):

Run the code script Main_GA_SVM.m 

Be sure that you revise the variable “Address” at the beginning the code. It should point to the directory that contain the feature vector “FV.xlsx” and targets “Targets.xlsx”

You can choose the kernel function as {1: linear, 2: gaussian, 3: polynomial, 4: rbf} with the variable “Kernels”

The variables “npop” and “max_generation” determine the number of population and maximum number of generations in genetic algorithm. 

The output variables are described as commented at the end of the code. The important output results will be saved at the directory “Address”.

4.	Genetic Algorithm based feature selection coupled with Decision Tree:

Run the code script Main_GA_DT.m 

Be sure that you revise the variable “Address” at the beginning the code. It should point to the directory that contain the feature vector “FV.xlsx” and targets “Targets.xlsx”

The variables “npop” and “max_generation” determine the number of population and maximum number of generations in genetic algorithm. 

The output variables are described as commented at the end of the code. The important output results will be saved at the directory “Address”.

5.	Genetic Algorithm based feature selection coupled with Logistic Regression:

Run the code script Main_GA_LR.m 

Be sure that you revise the variable “Address” at the beginning the code. It should point to the directory that contain the feature vector “FV.xlsx” and targets “Targets.xlsx”

The variables “npop” and “max_generation” determine the number of population and maximum number of generations in genetic algorithm. 

The output variables are described as commented at the end of the code. The important output results will be saved at the directory “Address”.

6.	Genetic Algorithm based feature selection coupled with FUZZY C-Means:

Run the code script Main_GA_FCM.m 

Be sure that you revise the variable “Address” at the beginning the code. It should point to the directory that contain the feature vector “FV.xlsx” and targets “Targets.xlsx”

The variables “npop” and “max_generation” determine the number of population and maximum number of generations in genetic algorithm. 

The output variables are described as commented at the end of the code. The important output results will be saved at the directory “Address”.

7.	Feature extraction form *.PNG femur images:

Run the code script Main_FE_V5.py. But note that the features extracted and are available in FV.xlsx file. Target.xlsx file is containing labels as well. 

References
1. https://www.mathworks.com/matlabcentral/fileexchange/74105-feature-selection-in-classification-using-genetic-algorithm?s_tid=FX_rc2_behav
for GA_based feature selection

2. https://github.com/pmneila/morphsnakes
for femur segmentation

