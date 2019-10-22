function labels = uselm_interface(X, C, L, lambda, hidden_dim, output_dim, normalize)
% Unsupervised ELM.
if nargin < 9
    selftune = false;
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
