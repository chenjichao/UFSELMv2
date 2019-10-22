function [SS, varargout] = ASC_llc2(X, C, eta, gamma, NN, iterations)


lambda = max(eta, 0.1);
gamma = max(gamma, 0);
NN = min(NN, size(X, 2)-1);

% build knn map
X = bsxfun(@minus, X, mean(X, 2));
% X = bsxfun(@rdivide, X, std(X, 0, 2));
num_samples = size(X, 2);
F = zeros(C, num_samples);
[knn_idx, knn_dist] = knnsearch(X', X', 'K', NN+1);
knn_idx = knn_idx(:, 2:end);
knn_dist = knn_dist(:, 2:end).^2;
[sorted_knn_idx, order] = sort(knn_idx, 2);
sorted_knn_dist = zeros(size(knn_dist));
for i = 1:size(knn_dist, 1)
    sorted_knn_dist(i, :) = knn_dist(i, order(i, :));
end
% TBD
% sorted_knn_dist = bsxfun(@minus, sorted_knn_dist, max(sorted_knn_dist, [], 2));
% sorted_knn_dist = exp(sorted_knn_dist);

S = zeros(num_samples);
for it = 1:iterations
    distF = l2dist(F, F);
    % update S via nnls
    for j = 1:num_samples
        idx = vec(sorted_knn_idx(j, :));
        dx = vec(sorted_knn_dist(j, :));
        df = vec(distF(idx, j));
        a = sqrt(eta*dx+gamma);
        vv = 0.5*lambda*df ./ a;
        lhs = [X(:, idx); diag(a)];
        rhs = [X(:, j); -vv];
        ss = lsqnonneg(lhs, rhs);
        S(idx, j) = ss;
    end
    % ========================================
    % From CAN, do not touch!
    % ========================================
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
% labels = spectral_clustering(SS, C);
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