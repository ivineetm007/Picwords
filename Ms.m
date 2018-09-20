function [Ims ,Kms,max] = Ms(I,bandwidth)

%% Mean Shift Segmentation (option: bandwidth)
I = im2double(I);
X = reshape(I,size(I,1)*size(I,2),3);                                         % Color Features
%% MeanShift
[clustCent,point2cluster,clustMembsCell] = MeanShiftCluster(X',bandwidth);    % MeanShiftCluster
max=0;
max2=0;
for i = 1:length(clustMembsCell)
    if(size(clustMembsCell{i},2)>max)
        max=size(clustMembsCell{i},2);
    end
end
for i = 1:length(clustMembsCell)
    if(size(clustMembsCell{i},2)>max2) && size(clustMembsCell{i},2)~=max 
        max2=size(clustMembsCell{i},2);
    end
end
%%For removing two largest cluster...
% for i = 1:length(clustMembsCell)                                              % Replace Image Colors With Cluster Centers
% if size(clustMembsCell{i},2)== max || size(clustMembsCell{i},2)== max2
%     X(clustMembsCell{i},:) =  repmat([1,1,1],size(clustMembsCell{i},2),1);
% else
%     X(clustMembsCell{i},:) = repmat(clustCent(:,i)',size(clustMembsCell{i},2),1);
% end
% end
%%For removing  largest cluster...
for i = 1:length(clustMembsCell)                                              % Replace Image Colors With Cluster Centers
    if size(clustMembsCell{i},2)== max 
        X(clustMembsCell{i},:) =  repmat([1,1,1],size(clustMembsCell{i},2),1);
    else
    X(clustMembsCell{i},:) = repmat(clustCent(:,i)',size(clustMembsCell{i},2),1);
    end
end
Ims = reshape(X,size(I,1),size(I,2),3);                                         % Segmented Image
Kms = length(clustMembsCell);

end
