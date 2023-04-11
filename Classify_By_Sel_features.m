%% Genetic Algorithm for Feature Selection in classification problems. 
clear all;close all;clc;
Address='H:\Projects_and_works\Finished Projects\MRI_Imaging\Revisions of the paper\V2\Osteoprosis_detection - V0\Scripts\Results\';
global orgfeatures labels alg Kernel
Sel_featurs=[1	2	5	8	9	11	16	30	36	38	40	44	45	46	47	48	50	51	60	63	64	66	67	69	72	73	75	77	79	80	84	96	100	104	106	115	119	121	123	125	126	133	135	136	144	147	148	152	159	161	162	166	168	172	174	180];
for i=1:10
    [num,txt,raw] = xlsread(append(Address,'FV.xlsx'));
    FV=num(:,1:end);
    [num,txt,raw] = xlsread(append(Address,'Targets.xlsx'));
    labels=num(:,1);
    features=F_Norm(FV);%Feature normalization
    features=features(:,Sel_featurs);
    orgfeatures=features;
    algorithms='SVM'; %{'KNN','NB','DT','NN','SVM'}
    Kernels={'linear','gaussian','polynomial','rbf'};
    Kernel=Kernels{1};
    Nf=size(features,2); % # of features
    [Accuracy_by_All,C]=fitf(1:Nf,Kernel);
    [stats] = statsOfMeasure(C,0);
    Acc(i,1)=stats{8,3};
    Fscore(i,1)=stats{9,3};
end
g1= repmat({'SVM_linrar'},size(Acc,1),1);

for i=1:10
    [num,txt,raw] = xlsread(append(Address,'FV.xlsx'));
    FV=num(:,1:end);
    [num,txt,raw] = xlsread(append(Address,'Targets.xlsx'));
    labels=num(:,1);
    features=F_Norm(FV);%Feature normalization
    features=features(:,Sel_featurs);
    orgfeatures=features;
    algorithms='SVM'; %{'KNN','NB','DT','NN','SVM'}
    Kernels={'linear','gaussian','polynomial','rbf'};
    Kernel=Kernels{4};
    Nf=size(features,2); % # of features
    [Accuracy_by_All,C]=fitf(1:Nf,Kernel);
    [stats] = statsOfMeasure(C,0);
    Acc(i,2)=stats{8,3};
    Fscore(i,2)=stats{9,3};
end
g2= repmat({'SVM_rbf'},size(Acc,1),1);

for i=1:10
    [num,txt,raw] = xlsread(append(Address,'FV.xlsx'));
    FV=num(:,1:end);
    [num,txt,raw] = xlsread(append(Address,'Targets.xlsx'));
    labels=num(:,1);
    features=F_Norm(FV);%Feature normalization
    features=features(:,Sel_featurs);
    orgfeatures=features;
    algorithms='SVM'; %{'KNN','NB','DT','NN','SVM'}
    Kernels={'linear','gaussian','polynomial','rbf'};
    Kernel=Kernels{3};
    Nf=size(features,2); % # of features
    [Accuracy_by_All,C]=fitf(1:Nf,Kernel);
    [stats] = statsOfMeasure(C,0);
    Acc(i,3)=stats{8,3};
    Fscore(i,3)=stats{9,3};
end
g3= repmat({'SVM_polynomial'},size(Acc,1),1);

for i=1:10
    [num,txt,raw] = xlsread(append(Address,'FV.xlsx'));
    FV=num(:,1:end);
    [num,txt,raw] = xlsread(append(Address,'Targets.xlsx'));
    labels=num(:,1)>0.5;
    features=F_Norm(FV);%Feature normalization
    features=features(:,Sel_featurs);
    orgfeatures=features;
    M = 200.0;
    maxIter = 100;
    minImprove = 1e-5;
    [centers,U] = fcm(features,2,[M maxIter minImprove true]);
    Choice1=U(1,:)'>0.5;
    Choice2=U(1,:)'<0.5;

    [Res,idx]=max([sum(labels==Choice1),sum(labels==Choice2)]);
    Acc(i,4)=Res/284;    
end
g4= repmat({'FCM'},size(Acc,1),1);

for i=1:10
    [num,txt,raw] = xlsread(append(Address,'FV.xlsx'));
    FV=num(:,1:end);
    [num,txt,raw] = xlsread(append(Address,'Targets.xlsx'));
    labels=num(:,1);
    features=F_Norm(FV);%Feature normalization
    features=features(:,Sel_featurs);
    orgfeatures=features;
    algorithms='DT'; %{'KNN','NB','DT','NN','SVM'}
    Kernels={'linear','gaussian','polynomial','rbf'};
    Kernel=Kernels{4};
    Nf=size(features,2); % # of features
    [Accuracy_by_All,C]=fitf_DT(1:Nf,Kernel);
    [stats] = statsOfMeasure(C,0);
    Acc(i,5)=stats{8,3};
    Fscore(i,5)=stats{9,3};
end
g5= repmat({'Decision Tree'},size(Acc,1),1);

for i=1:10
    [num,txt,raw] = xlsread(append(Address,'FV.xlsx'));
    FV=num(:,1:end);
    [num,txt,raw] = xlsread(append(Address,'Targets.xlsx'));
    labels=num(:,1);
    features=F_Norm(FV);%Feature normalization
    features=features(:,Sel_featurs);
    orgfeatures=features;
    algorithms='DT'; %{'KNN','NB','DT','NN','SVM'}
    Kernels={'linear','gaussian','polynomial','rbf'};
    Kernel=Kernels{4};
    Nf=size(features,2); % # of features
    [Accuracy_by_All,C]=fitf_LR(1:Nf,Kernel);
    [stats] = statsOfMeasure(C,0);
    Acc(i,6)=stats{8,3};
    Fscore(i,6)=stats{9,3};
end
g6= repmat({'Logistic Regression'},size(Acc,1),1);

g= [g1;g2;g3;g4;g5;g6];
boxplot(Acc,g);grid on;    