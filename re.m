clear
clc


addpath(genpath('data'));
addpath(genpath('graph'));
[data, gnd] = load_data('WebKBTexas');
[nSmp, nFtr] = size(data);
X = normalize(data, 'range', [-1,1]); % same as mapminmax but to columns
% X = normalize(data,'center','mean');
% X = mapminmax(data, -1, 1); % by y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin;
Y = gnd;
nCls = length(unique(Y));
% X = [nSmp, nFtr]
% Y = [nSmp, 1]

% permdata = 0
% if permdata
%     perm = randperm(size(X, 1));
%     X = X(perm, :);
%     Y = Y(perm, :);
% end

% [y, ~] = litekmeans(X, nCls, 'MaxIter', 100);


nNbr = 10;
scale = 0;
lambda = 1;
nNrn = 1024;
nOut = 32;

options.NN = nNbr;
options.GraphDistanceFunction = 'euclidean';
options.GraphWeights = 'distance';
options.LaplacianNormalize = 1;
options.LaplacianDegree = 1;
options.GraphWeightParam = 1;
L = laplacian(options, X);
normalize = 1;

y = uselm_interface(X', nCls, L, lambda, nNrn, nOut, normalize); 

uselm_interface use kmeans instead of litekmeans

ACC = accuracy(Y, y)
results = zeros(3, 1);
results = ClusteringMeasure(Y, y)
% NMI = nmi(gnd, y)*100;
