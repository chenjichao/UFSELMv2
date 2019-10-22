function [beta, F] = UFSELM(origdata, L, nCls, lambda1, lambda2, nNrn)
% Unsupervised Feature Selection based Extreme Learning Machine
warning('off','all');

itermax = 10;

%% Init
[nSmp, nFtr] = size(origdata);

%%%%%%%%%%% Random generated input weights input_weight and 
%%%%%%%%%%% biases bias (b_i) of hidden neurons
input_weight = rand(nFtr, nNrn) * 2 - 1;
bias = rand(1, nNrn);
tempH = bsxfun(@plus, origdata*input_weight, bias);
%%%%%%%%%%% Calculate hidden neuron output matrix H with sig ActFun
H = 1 ./ (1 + exp(-tempH));
%%%% Centralization
H = normalize(H,'center','mean');
% H [nSmp, nFtr]

%%%%%%%% Initialize cluster indicator matrix F[nSmp,nCls] by kmeans on the original data
StartInd = randsrc(nSmp, 1, 1:nCls);
[res_km, ~, ~] = kmeans_ldj(origdata, StartInd, 1);
F = zeros(nSmp, nCls);
% F: cluster indicator matrix F [nSmp, nCls]
for i = 1:nCls
    F((res_km==i), i) = 1;
end
%%%%%%% Input F as scaled cluster indicator
F = F*((F'*F)^(-1/2));
%%%%%%% Init B as an identity matrix and compute Q
B = eye(nNrn);
Q = (H'*H + lambda1*B + lambda2*H'*L*H);

%% Repeat
for iter = 1:itermax
%     iter
    %%%% Update beta ------------------------------------------------------
    beta = Q \ (H'*F);
    %%%% Update B ---------------------------------------------------------
%     for i = 1:size(beta,2)
%         B(i, i) = (norm(beta(i,:))+1e-30)^-1;
%     end
    Q = (H'*H + lambda1*B + lambda2*H'*L*H);
    %%%% Calculate P ------------------------------------------------------
%     P = eye(nSmp) - 2*H/Q*H' + (H/Q)*((H'*H) + lambda1*B + lambda2*H'*L*H)*(Q\H');
%     P = eye(nSmp) - 2*H/Q*H' + (H/Q)*(Q)*(Q\H');
    P = eye(nSmp) - 2*H/Q*H';

    %%%% Update F ---------------------------------------------------------
    F = eig_decom_sa(P,nCls);

    Z = zeros(nSmp,nCls);
    Z(F == max(F,[],2)) = 1;
    F = Z;
%     F = F*((F'*F+1e-30)^(-1/2));
    
%     loss = norm((H*beta - F), 'fro') + lambda1*trace(beta'*B*beta) + ...
%         lambda2*trace(beta'*H'*L*H*beta)
end
%     [~,y] = max(F,[],2);
%     ACC = accuracy(gnd, y);
end
