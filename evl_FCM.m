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


function [out,C]=evl_FCM(FV,labels,Kernel)
%% ===================fitness function based on SVM==============================
labels=labels>0.5;
%features=F_Norm(FV);%Feature normalization

[centers,U] = fcm(FV,2);
Choice1=U(1,:)'>0.5;
Choice2=U(1,:)'<0.5;

[Res,idx]=max([sum(labels==Choice1),sum(labels==Choice2)]);
% Acc=Res/284;
% out=100*Acc;
if idx==1
    elabel=Choice1;
else
    elabel=Choice2;
end
C = confusionmat(labels,elabel);
acc1=C(1,1)/sum(C(1,:),"all");
acc2=C(2,2)/sum(C(2,:),"all");
Acc=mean([acc1,acc2]);
NF=size(FV,2);
out=100*Acc+1/NF;




