%% Genetic Algorithm for Feature Selection in classification problems. 
clear all;close all;clc;
Address='C:\Users\Marab\Desktop\Osteoprosis_detection\Scripts\Results\';


%%
global orgfeatures labels alg Kernel
%load DB.mat
Tstart=tic;
[num,txt,raw] = xlsread(append(Address,'FV.xlsx'));
FV=num(:,1:end);
[num,txt,raw] = xlsread(append(Address,'Targets.xlsx'));
labels=num(:,1);
features=F_Norm(FV);%Feature normalization
orgfeatures=features;
Nf=size(features,2); % # of features
%% initialization:
algorithms='SVM'; %{'KNN','NB','DT','NN','SVM'}
Kernels={'linear','gaussian','polynomial','rbf'};
Kernel=Kernels{3};
[Accuracy_by_All,C_All]=fitf(1:Nf,Kernel);

[stats] = statsOfMeasure(C_All, 1);
Tstop=toc(Tstart);