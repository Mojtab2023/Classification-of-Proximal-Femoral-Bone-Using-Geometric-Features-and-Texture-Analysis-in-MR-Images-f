
%% Genetic Algorithm for Feature Selection in classification problems. 
clear all;close all;clc;
Address='H:\Projects_and_works\Finished Projects\MRI_Imaging\Revisions of the paper\V2\Osteoprosis_detection - V0\Scripts\Results\';


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
%% initialization:
algorithms='SVM'; %{'KNN','NB','DT','NN','SVM'}
Kernels={'linear','gaussian','polynomial','rbf'};
Kernel=Kernels{3};
npop=100; % initial population size
max_generation=500;
%%
Nf=size(features,2); % # of features
Best_solution=zeros(max_generation,Nf);
alg=algorithms;
disp(append(algorithms,'_',Kernel,' have been just started'));
for i=1:npop
    solutions{i}=make_solution(Nf);
end
gen=1;
for t=1:max_generation
    t
    for i=1:npop
        [fitness(i),Ctmp{i}]=fitf(solutions{i},Kernel);
    end
    [Cost(gen),idxtmp]=max(fitness);%cost function of the best individual in each generation
    Cgen{gen}=Ctmp{idxtmp};%confusion matrix of best individual (that reached the max fitness) in each generation
    
    if rem(t,10)==0
        disp(['Searching...',num2str(t/max_generation*100),'%  , accuracy= ',num2str(max(fitness))]);
    end
    [~,idx]=sort(fitness);

    best_solution=solutions{idx(end)};
    best_solution2=solutions{idx(end-1)};
    solutions{idx(1)}=GASearch(best_solution,best_solution2,Nf,Kernel); 
    solutions{idx(2)}=GASearch(best_solution2,best_solution,Nf,Kernel);
    Best_solution(gen,1:length(best_solution))=best_solution;%selected features of the best individual in each generation
    gen=gen+1;
end
disp(append(algorithms,'_',Kernel,' have been done'));

[Accuracy_by_All,C_All]=fitf(1:Nf,Kernel);
[Accuracy_best,indx_best]=max(Cost);
BFV=Best_solution(indx_best,:);%best feature vector
BFV = BFV(find(BFV,1,'first'):find(BFV,1,'last'));%remove zero elements
BC=Cgen{indx_best};%best conf matrix
[stats] = statsOfMeasure(BC, 1);
Tstop=toc(Tstart)
save(append(Address,algorithms,'_',Kernel,'.mat'),'Cost','Tstop',"Best_solution",'BFV','stats','Accuracy_by_All','C_All')








