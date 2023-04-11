% function out=evl(data,labels,algorithm)
%     original_data=data;
%     original_label=labels;
%     ten=round(size(data,1)/10);
%     for i=1:10   
%         data=original_data;
%         label=original_label;
%         test=data((i-1)*ten+1:i*ten,:);
%         l_test=label((i-1)*ten+1:i*ten,:);
%         data((i-1)*ten+1:i*ten,:)=[];
%         label((i-1)*ten+1:i*ten,:)=[];
%         train=data;
%         l_train=label;
%         switch algorithm
%             case 'NB'
%                 Class = NB(train,l_train,test);
%             case 'KNN'
%                 Class= KNN(train,l_train,test);
%             case 'DT'
%                 Class= DT(train,l_train,test);
%             case 'NN'
%                 Class=NN(train,l_train,test); 
%         end
%         accuracy(i)= evaluation(Class,l_test);
%     end
%     out=mean(accuracy)*100;
%     
% end


function [out,C]=evl_LR(FV,labels,Kernel)
%% ===================fitness function based on SVM==============================

n = size(FV,1); % Number of observations
cvp = cvpartition(n,'KFold',10);
discrCVModel = fitclinear(FV,labels,'CVPartition',cvp);
Acc=1-kfoldLoss(discrCVModel);


NF=size(FV,2);
out=100*Acc+1/NF;
[elabel,escore] = kfoldPredict(discrCVModel);

C = confusionmat(labels,elabel);