function beta = rayleigh_ritz(A, H, K)
% Solve the Rayleigh-Ritz problem:
% min_{beta} tr(beta'*A*beta)
% s.t. beta'*H*H'*beta=I.
%
% Assume that A is positive definite, and B=H*H' is positive semi-definite.
A = (A+A')/2;
[L, N] = size(H);
if L < N  % full rank
    B = H*H';
    [V, D] = eig(A, B);
    [~, idx] = sort(real(diag(D)), 'ascend');
    beta = real(V(:, idx(1:K)));
    for jj = 1:K
        beta(:, jj) = beta(:, jj) / norm(H'*beta(:, jj));
    end
else
    [U, S, V] = svd(H, 'econ');
    S = diag(S);
    pos = sort(find(S>1e-8));
    M = diag(1./S(pos))*U(:, pos)'*A*U(:, pos)*diag(1./S(pos));
    M = (M+M')/2;
    [VV, DD] = eig(M);
    [~, idx]= sort(real(diag(DD)), 'ascend');
    alpha = zeros(N, K);
    alpha(1:length(pos), :) = diag(1./S(pos))*real(VV(:, idx(1:K)));
    beta = U*alpha;
end


end