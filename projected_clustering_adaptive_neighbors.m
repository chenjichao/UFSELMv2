function [labels, varargout] = projected_clustering_adaptive_neighbors(X, C, NN, output_dim, iterations)
% Projected clustering with adaptive neighbors.
% The objective function is
% min_{S,F,W} tr(W'*X*Ls*X'*W)+gamma*||S||^2 + lambda*tr(F*Ls*F')
%     s.t. W'*St*W = I
%
num_samples = size(X, 2);
S = zeros(num_samples);
% F = zeros(C, num_samples);
distX = l2dist(X, X);
[distX1, idx] = sort(distX, 2);
gammas = zeros(num_samples, 1);
for i = 1:num_samples
    di = distX1(i, 2:NN+2);
    gammas(i) = 0.5*(NN*di(end)-sum(di(1:end-1)));
    S(i, idx(i, 2:NN+2)) = (di(end)-di)/(NN*di(end)-sum(di(1:end-1))+eps);
end
gamma = mean(gammas);
lambda = gamma;
SS = (S+S')/2;
Ls = diag(sum(SS, 2)) - SS;
Xm = bsxfun(@minus, X, mean(X, 2));
% invSt = inv(X*(eye(num_samples)-ones(num_samples)/num_samples)*X');
% W = eig1(X*Ls*X', output_dim, 0, 0);
F = eig1(Ls, C, 0)';

for it = 1:iterations
    % update W
    W = rayleigh_ritz(X*Ls*X', Xm, output_dim);
    
    distX = l2dist(W'*X, W'*X);
    distF = l2dist(F, F);
    % Update S
    for j = 1:num_samples
        d = distX(:, j) + lambda*distF(:, j);
        s = EProjSimplex_new(-d/2/gamma);
        S(:, j) = s(:);
    end
    % Update F
    SS = (S + S')/2;
    Ls = diag(sum(SS, 2)) - SS;
    [V, D] = eig(Ls);
    
    D = diag(D);
    [~, idx] = sort(D, 'ascend');
    F_old = F;
    F = V(:, idx(1:C))';
    % update lambda
    fn1 = sum(D(idx(1:C)));
    fn2 = sum(D(idx(1:C+1)));
    if fn1 > 1e-11
        lambda = 2 * lambda;
    elseif fn2 < 1e-11
        lambda = lambda / 2;
        F = F_old;
    else
        break;
    end
end
[C1, ~] = conncomp(sparse(SS));
if C1 ~= C
    warning('Got %d components while class number is %d', C1, C);
    flag = 0;
else
    flag = 1;
end
labels = spectral_clustering(SS, C);
if nargout > 1
    varargout{1} = flag;
end
end

function dist = l2dist(A, B)
% Compute the distance between each column of A and B.
As = sum(A.^2, 1);
Bs = sum(B.^2, 1);
dist = repmat(As', [1, size(B, 2)]) + repmat(Bs, [size(A, 2), 1]) - 2*A'*B;
dist = max(real(dist), 0);
end
