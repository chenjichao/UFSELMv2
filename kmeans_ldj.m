function [Ind, sumd, center] = kmeans_ldj(M, StartIndMeanK, isVector)
% each row is a data point
%{
Input:  
        M: the data points [nSample, nFeature]
        StartIndMeanK: the cluster indicator of data [nSample, 1]
        isVector: boolean, isvector(StartIndMeanK)
Output: 
        Ind: cluster indicator of data [1, nSample]
        sumd: the sum of distance the data within clusters and the centroid
        center: the centroids [nCluster, nFeature]
%}

[nSample, nFeature] = size(M);

if isscalar(StartIndMeanK)
    StartIndMeanK = randsrc(nSample, 1, 1:StartIndMeanK);
end

if isvector(StartIndMeanK) && isVector
    K = length(StartIndMeanK);
    if K == nSample
        K = max(StartIndMeanK);
        means = zeros(K, nFeature);
        for ii=1:K
            means(ii,:) = mean(M(find(StartIndMeanK==ii),:),1);
        end
    else
        means = zeros(K,nFeature);
        for ii=1:K
            means(ii,:) = M(StartIndMeanK(ii),:);
        end
    end
else
    K = size(StartIndMeanK,1);
    means = StartIndMeanK;
end









center = means; % the mean of each cluster [nCluster, nFeature]
M2 = sum(M.*M, 2)';
twoMp = 2*M';
M2b = repmat(M2,[K,1]);
Center2 = sum(center.*center,2);
Center2a = repmat(Center2,[1,nSample]);
[~, Ind] = min(abs(M2b + Center2a - center*twoMp));
% nearest centroid for each sample [1, nSample]
Ind2 = Ind;
it = 1;
%while true
while it < 200
    for j = 1:K
        dex = find(Ind == j);
        l = length(dex);
        if l > 1;   center(j,:)=mean(M(dex,:));
        elseif l == 1;  center(j,:)=M(dex,:);
        else;   t=randperm(nSample);center(j,:)=M(t(1),:);
        end
    end
    Center2 = sum(center.*center,2);
    Center2a = repmat(Center2,[1,nSample]);
    [dist, Ind] = min(abs(M2b + Center2a - center*twoMp));
    % nearest centroid for each sample [1, nSample]
    if Ind2==Ind;   break;  end
    Ind2 = Ind;
    it = it+1;
end
sumd = zeros(K,1);
for ii=1:K
    idx = Ind==ii;
    dist2 = dist(idx);
    sumd(ii) = sum(dist2);
end
end
