function [labels, varargout] = clustering_adaptive_neighbors(X, C, NN, iterations)
% Clustering with adaptive neighbors.
% The objective function is
% min_{S,F} tr(X*Ls*X') + gamma*||S||^2 + lambda*tr(F*Ls*F').
num_samples = size(X, 2);
S = zeros(num_samples);
F = zeros(C, num_samples);
distX = l2dist(X, X);
[distX1, ~] = sort(distX, 2);
gammas = zeros(num_samples, 1);
for i = 1:num_samples
    di = distX1(i, 2:NN+2);
    gammas(i) = 0.5*(NN*di(end)-sum(di(1:end-1)));
end
gamma = mean(gammas);
lambda = gamma;
for it = 1:iterations+1
    distF = l2dist(F, F);
    % Update S
    for j = 1:num_samples
        d = distX(:, j) + lambda*distF(:, j);
        s = EProjSimplex_new(-d/2/gamma);
        S(:, j) = s(:)';
    end
    % Update F
    SS = (S + S')/2;
    Ls = diag(sum(SS, 2)) - SS;
    [V, D] = eig(Ls);
    D = diag(D);
    [~, idx] = sort(D, 'ascend');
    F = V(:, idx(1:C))';
    % update lambda
    fn1 = sum(D(idx(1:C)));
    fn2 = sum(D(idx(1:C+1)));
    if fn1 > 1e-11
        lambda = 2 * lambda;
    elseif fn2 < 1e-11
        lambda = lambda / 2;
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
dist = max(dist, 0);
end
