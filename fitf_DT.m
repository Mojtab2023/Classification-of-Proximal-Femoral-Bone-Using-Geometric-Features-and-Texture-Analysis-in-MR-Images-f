function [acc,C]=fitf_DT(sub,Kernel)
    global orgfeatures labels alg
    features=orgfeatures(:,sub);
%     acc=evl(features,labels,alg);
    [acc,C]=evl_DT(features,labels,Kernel);
end