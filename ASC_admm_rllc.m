function [SS, varargout] = ASC_admm_rllc(X, C, eta, gamma, NN, iterations)
% Adaptive subspace clustering with robust local linear constraints:
%
% min_{S} ||X-XS||_{2,1}+(eta/2)*sum||d_i \odot s_i||^2
% s.t. S_ii = 0, S_ij >= 0.
%
% The problem is solved via ADMM:
%
% (for each s in columns of S)
% min ||e||_2 + (eta/2)*||dx \odot g||^2 + (rho/2)||Xg-e-x+u||^2 +
% (rho/2)||g-s+v||^2 + lambda*df'*s
% s.t. s>=0.


lambda = max(eta, 0.1);
NN = min(NN, size(X, 2)-1);
gamma = max(gamma, 0);

X = bsxfun(@minus, X, mean(X, 2));
num_samples = size(X, 2);
F = zeros(C, num_samples);
% build knn map
[knn_idx, knn_dist] = knnsearch(X', X', 'K', NN+1);
knn_idx = knn_idx(:, 2:end);
knn_dist = knn_dist(:, 2:end).^2;
[sorted_knn_idx, order] = sort(knn_idx, 2);
sorted_knn_dist = zeros(size(knn_dist));
for i = 1:size(knn_dist, 1)
    sorted_knn_dist(i, :) = knn_dist(i, order(i, :));
end
% sorted_knn_dist = bsxfun(@minus, sorted_knn_dist, max(sorted_knn_dist, [], 2));
% sorted_knn_dist = exp(sorted_knn_dist).^2;

% init S
S = zeros(num_samples);
for i = 1:num_samples
    S(sorted_knn_idx(i, :), i) = 1;%/NN;
%     S(sorted_knn_idx(i, :), i) = lsqnonneg(X(:, sorted_knn_idx(i, :)), X(:, i));
end
SS = (S+S')/2;
Ls = diag(sum(SS, 2)) - SS;
F = eig1(Ls, C, 0)';

% ADMM parameters
itADMM = 10;
% NOTE: This impl minimizes each column of S via a separate ADMM routine.
for it = 1:iterations
    distF = l2dist(F, F);
    % minimize each column of S
    for j = 1:num_samples
        idx = vec(sorted_knn_idx(j, :));
        dx2 = eta*vec(sorted_knn_dist(j, :))+gamma*ones(NN, 1);
        df = lambda*vec(distF(idx, j));
        ss = solve_s_admm(X(:, j), X(:, idx), dx2, df, S(idx, j), itADMM);
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


%% Solve the following problem via ADMM:
% min||x-Ds||_2 + (1/2)dx^2 \dot s^2 + df'*s
% s.t. s >= 0.
function s = solve_s_admm(x, D, dx2, df, s0, iterations)
rho = 10;
abs_tol = 1e-3;
rel_tol = 1e-3;
tau = 2;
mu = 10;
alpha = 1.8;
e = D*s0-x;
s = s0;
u = zeros(size(D, 1), 1);
v = zeros(size(s0));
DTDrI = D'*D+eye(size(D, 2));
ddx2 = diag(dx2);
opt.POSDEF = true; opt.SYM = true;
for it = 1:iterations
    sprev = s;
    eprev = e;
    % update g
%     g = (DTDrI+diag(dx2)/rho)\(D'*(e+x-u)+s-v-df/rho);
    g = linsolve(DTDrI+ddx2/rho, D'*(e+x-u)+s-v-df/rho, opt);
    % over-relaxation
    Dg = D*g;
    Dgor = alpha*Dg + (1-alpha)*(e+x);
    gor = alpha*g + (1-alpha)*s;
    % update e
    e = prox_l2(Dgor-x+u, 1/rho);
    % update s
    s = max(gor+v, 0);
    % dual variable update
    u = u + Dgor - e - x;
    v = v + gor - s;
    % stopping criteria
    nAX = sqrt(sum(Dg.^2)+sum(g.^2));
    nBZ = sqrt(sum(e.^2)+sum(s.^2));
    nC = sqrt(sum(x.^2));
    nATY = norm(D'*u+v);
    nR = sqrt(sum((Dg-e-x).^2)+sum((g-s).^2));
    nS = rho*norm(D'*(e-eprev)+s-sprev);
    epri = sqrt(size(D,1)+size(D,2))*abs_tol + max([nAX,nBZ,nC])*rel_tol;
    edua = sqrt(size(D,2))*abs_tol + rho*nATY*rel_tol;
    if nR > mu*nS
        rho = rho * tau;
        u = u / tau;
        v = v / tau;
    elseif nS > mu*nR
        rho = rho / tau;
        u = u * tau;
        v = v * tau;
    end
    if (nR < epri) && (nS < edua)
        break;
    end
%     fprintf('Iter %02d: r: %.3f, epri: %.3f, s: %.3f, edua: %.3f, rho: %.3f\n', it, nR, epri, nS, edua, rho);
end
end

%% L2 prox operator
function out = prox_l2(x, lambda)
xn = norm(x);
out = max(xn-lambda, 0) * x / xn;
end

%% L2dist
function dist = l2dist(A, B)
% Compute the distance between each column of A and B.
As = sum(A.^2, 1);
Bs = sum(B.^2, 1);
dist = repmat(As', [1, size(B, 2)]) + repmat(Bs, [size(A, 2), 1]) - 2*A'*B;
dist = max(dist, 0);
end