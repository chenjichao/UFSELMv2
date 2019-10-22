function L = Adjacency2Laplacian(A, normalize)


if nargin < 2
    normalize = true;
end

D = sum(A, 2);

if normalize
    D(D~=0) = sqrt(1./D(D~=0));
    D = spdiags(D, 0, speye(size(A, 1)));
    A = D*A*D;
    L = speye(size(A, 1)) - A; % L = I - D^-1/2*A*D^-1/2
else
    L = spdiags(D, 0, speye(size(A, 1))) - A; % L = D - W
end