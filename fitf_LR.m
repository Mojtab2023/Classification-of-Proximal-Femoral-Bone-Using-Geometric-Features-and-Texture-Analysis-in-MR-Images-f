function [acc,C]=fitf_LR(sub,Kernel)
    global orgfeatures labels alg
    features=orgfeatures(:,sub);
%     acc=evl(features,labels,alg);
    [acc,C]=evl_LR(features,labels,Kernel);
end