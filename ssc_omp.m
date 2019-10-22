function W = ssc_omp(X, epsilon, L, verbose)
% Compute connectivity matrix using SSC-OMP.
params.numThreads = 1;
if isempty(epsilon)
    params.L = L;
else
    params.eps = epsilon;
end
X = normcols(X);
A = cell(1, size(X, 2));
t = tic;
for i = 1:size(X, 2)
    Xi = X;
    Xi(:, i) = 0;  % To be determined; this should be safe.
    ai = mexOMP(X(:, i), Xi, params);
    A{i} = ai / max(abs(ai));
    if verbose && toc(t) > 5
        fprintf('Processing %d/%d samples...\n', i, size(X, 2));
        t = tic;
    end
end
A = cat(2, A{:});
% A = bsxfun(@rdivide, A, max(A, [], 1));
W = abs(A) + abs(A)';
end