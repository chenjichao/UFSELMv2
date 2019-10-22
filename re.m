clear
clc



[data, gnd] = load_data('glass');
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

[y, ~] = litekmeans(X, nCls, 'MaxIter', 100);

ACC = accuracy(Y, y)
results = zeros(3, 1);
results = ClusteringMeasure(Y, y)
% NMI = nmi(gnd, y)*100;