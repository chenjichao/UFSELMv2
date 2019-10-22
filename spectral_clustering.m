function labels = spectral_clustering(W, K, normalize, neigs)
if nargin < 3
    normalize = true;
end
if nargin < 4
    neigs = K;
end

% compute laplacian
D = sum(W, 2);
if normalize
    D(D~=0)=sqrt(1./D(D~=0));
    D=spdiags(D,0,speye(size(W,1)));
    W=D*W*D;
    L=speye(size(W,1))-W; % L = I-D^-1/2*W*D^-1/2
else
    L = spdiags(D,0,speye(size(W,1)))-W; % L = D-W
end
% compute eigenvectors
[U, d] = eig(full(L));
[~, idx] = sort(real(diag(d)));
U = real(U(:, idx(1:neigs)));
if normalize
    U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2))+eps);
end
labels = kmeans(U, K, 'Replicates', 100, 'Start', 'plus', 'MaxIter', 1000);

end