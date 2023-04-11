%% Genetic Algorithm for Feature Selection in classification problems. 
clear all;close all;clc;
Address='C:\Users\Marab\Desktop\Osteoprosis_detection\Scripts\Results\';

Tstart=tic;
[num,txt,raw] = xlsread(append(Address,'FV.xlsx'));
FV=num(:,1:end);
% FV2=num(:,1:end).^2;
% FV3=num(:,1:end).^3;
% FV=cat(2,FV1,FV2,FV3);
[num,txt,raw] = xlsread(append(Address,'Targets.xlsx'));
labels=num(:,1)>0.5;
features=F_Norm(FV);%Feature normalization

M = 200.0;
maxIter = 100;
minImprove = 1e-5;
[centers,U] = fcm(features,2,[M maxIter minImprove true]);
Choice1=U(1,:)'>0.5;
Choice2=U(1,:)'<0.5;

[Res,idx]=max([sum(labels==Choice1),sum(labels==Choice2)]);
Acc=Res/284;

if idx==1
    elabel=Choice1;
else
    elabel=Choice2;
end
C = confusionmat(labels,elabel);

RC(:,1)=labels;
RC(:,2)=elabel;
RC(:,3)=(elabel==labels);

[stats] = statsOfMeasure(C, 1);