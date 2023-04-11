function [acc,C]=fitf(sub,Kernel)
    global orgfeatures labels alg
    features=orgfeatures(:,sub);
%     acc=evl(features,labels,alg);
    [acc,C]=evl(features,labels,Kernel);
end