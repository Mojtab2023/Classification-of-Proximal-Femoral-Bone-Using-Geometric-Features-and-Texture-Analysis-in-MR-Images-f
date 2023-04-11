function [acc,C]=fitf_FCM(sub,Kernel)
    global orgfeatures labels alg
    features=orgfeatures(:,sub);
%     acc=evl(features,labels,alg);
    [acc,C]=evl_FCM(features,labels,Kernel);
end