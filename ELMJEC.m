function [G] = ELMJEC(origdata, L,nclass, lambda, lambda2, Red_dim, nNrn)
% function [G, Q, F, input_weight, bias] = ELMJEC(origdata, L, Red_dim, ...
%     nclass, lambda, lambda2, nNrn)

% G is the cluster indicator
% Q is the ELM output weights
% F is the cluster centroid vector

% Red_dim: reduced dim, embedding dimension

%%%%%%%%%%% Random generate input weights input_weight and 
%%%%%%%%%%% biases bias (b_i) of hidden neurons
origdata = origdata';
X = origdata;
input_weight = rand(size(X, 1), nNrn) * 2 - 1;
bias = rand(1, nNrn);
tempH = X' * input_weight;
clear X;
tempH = bsxfun(@plus, tempH, bias);

%%%%%%%%%%% Calculate hidden neuron output matrix H with sig ActFun
H = 1 ./ (1 + exp(-tempH));
clear tempH;
data = H';
[~, nSmp] = size(data);
intermax = 20;

%%%% Centralization
data = data - repmat(mean(data, 2), 1, size(data, 2));

%%%%%%%% Initialize G based on kmeans on the original data
StartInd = randsrc(nSmp, 1, 1:nclass);
[res_km, error, ~] = kmeans_ldj(origdata', StartInd, 1);
G = zeros(nSmp, nclass);
for cn = 1:nclass
    G((res_km==cn), cn) = 1;
end
clear res_km



%%%%%%%%% Fix G to update Q and F
temp = diag(1 ./ diag(G' * G));
M = (1 - lambda) * (eye(nSmp)) + lambda * G * temp * G' - lambda2 * L;
P = data*M*data';

% Q = eig_decom(data * M * data', Red_dim);
Q = eig_decom(P, Red_dim);

% data is H' in the paper
% data [nFeature, nSample]

F = Q' * data * G * temp;


%%%% Main Run
for ii=1:intermax
    %%%%%%%%% Fix Q to compute G
    XW = data' * Q;
    
    [index, sumd, ~] = kmeans_ldj(XW, F', 0);
    error = sum(sumd);
    indicator = 0;
    ntime = 1;
    while indicator == 0 && ntime < 20
        StartInd = randsrc(nSmp, 1, 1:nclass);
        [res_km, sumd, ~] = kmeans_ldj(XW, StartInd, 1);
        obj_km = sum(sumd);
        if obj_km < error
            indicator = 1;
        end
        ntime = ntime + 1;
    end
    if ntime == 20
        res_km = index;
    end
    
    G_old = G;
    G = zeros(nSmp, nclass);
    for cn = 1:nclass
        G((res_km == cn), cn) = 1;
    end
    
    %%%%%%%%% Fix G to update Q and F
    temp = diag(1 ./ (diag(G' * G)));
    M = (1 - lambda) * (eye(nSmp)) + lambda * G * temp * G' - lambda2 * L;

    P = data*M*data';
    Q = eig_decom(P, Red_dim);
    
    F = Q' * data * G * temp;
    
    %%%%%%%%% Check stop condition
    if  isequal(G, G_old)
        break;
    end
end

end