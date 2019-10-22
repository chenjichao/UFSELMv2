function [SS, varargout] = ASC_admm_l2_local(X, C, eta, gamma, NN, iterations)
% Adaptive subspace clustering that solves the following problem with local constraints:
% min (1/2)||X-XS||^2 + eta*sum_ij||x_i-x_j||^2 s_ij
%
% The problem is solved via ADMM:
% min (1/2)||X-XG||^2 + eta*sum_{ij}||x_i-x_j||^2 s_ij +
% (rho/2)||G-S+U||^2


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
S = zeros(num_samples);
for i = 1:num_samples
    S(sorted_knn_idx(i, :), i) = 1/NN;
end
lambda = max(eta, 0.1);
gamma = max(gamma, 0);

% ADMM parameters
rho = 10;
abs_tol = 1e-3;
rel_tol = 1e-3;
tau = 2;
mu = 10;
G = zeros(size(S));
U = zeros(size(S));
for it = 1:iterations
    distF = l2dist(F, F);
    % update S via ADMM
    Sold = S;
    for j = 1:num_samples
        % G subproblem
        idx = vec(sorted_knn_idx(j, :));
        XTXrI = X(:, idx)'*X(:, idx) + (rho+gamma)*eye(NN);
        G(idx, j) = XTXrI\(X(:, idx)'*X(:, j)+rho*(S(idx, j)-U(idx, j)));
        % S subproblem
        dx = vec(sorted_knn_dist(j, :));
        df = vec(distF(idx, j));
        d = G(idx, j) + U(idx, j) - (eta*dx+lambda*df)/rho;
        S(idx, j) = max(d, 0);
    end
    U = U + G - S;
    % stopping criteria for admm
    nG = norm(G(:));
    nS = norm(S(:));
    nU = norm(U(:));
    r = norm(vec(S-G));
    s = norm(vec(rho*(Sold-S)));
    epri = sqrt(NN*num_samples)*abs_tol + max(nG, nS)*rel_tol;
    edua = sqrt(NN*num_samples)*abs_tol + rho*nU*rel_tol;
    if r > mu*s
        rho = rho * tau;
        U = U / tau;
    elseif s > mu*r
        rho = rho / tau;
        U = U * tau;
    end
%     fprintf('Iter %02d: r: %.3f, epri: %.3f, s: %.3f, edua: %.3f, rho: %.3f\n', it, r, epri, s, edua, rho);
    if (r < epri) && (s < edua)
        break;
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
%     fprintf('Average nnz of S: %g\n', mean(sum(S~=0, 1)));
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
