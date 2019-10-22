function labels = uselm_interface(X, C, NN, scale, lambda, hidden_dim, output_dim, normalize, selftune)
% Unsupervised ELM.
if nargin < 9
    selftune = false;
end
% build laplacian
if selftune
    [W, ~] = selftuning(X', NN);
    D = sum(W, 2);
    L = spdiags(D,0,speye(size(W,1)))-W;
else
    options = struct('NN', NN, 'GraphDistanceFunction', 'euclidean', ...
        'GraphWeights', 'heat', 'GraphWeightParam', scale, ...
        'LaplacianNormalize', normalize, 'LaplacianDegree', 1);
    L = laplacian(options, X');
end
% run US-ELM
params = struct('NE', output_dim, 'NumHiddenNeuron', hidden_dim, ...
    'NormalizeInput', 0, 'NormalizeOutput', normalize, 'Kernel', 'sigmoid', ...
    'lambda', lambda);
uselm_model = uselm(X', L, params);
% run KMeans
% At some occasions the Embed is complex. Simply fail the algorithm.
if isreal(uselm_model.Embed)
    labels = kmeans(uselm_model.Embed, C, 'Replicates', 100, ...
        'Start', 'plus', 'Display', 'off');
else
    labels = [];
end
end