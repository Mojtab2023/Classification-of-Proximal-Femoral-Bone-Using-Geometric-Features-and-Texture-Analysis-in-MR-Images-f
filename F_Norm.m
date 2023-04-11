function FV=F_Norm(FV)
[x,y]=size(FV);
cnt=0;
flag=0;
for i=1:y
    FV(:,i)=FV(:,i)-mean(FV(:,i));
    if var(FV(:,i))~=0
        FV(:,i)=FV(:,i)/var(FV(:,i));
    else
        cnt=cnt+1;
        Deletlist(cnt)=i;
        flag=1;
    end
end
if flag==1
    FV(:,Deletlist)=[];
end