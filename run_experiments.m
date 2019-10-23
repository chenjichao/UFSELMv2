function varargout = run_experiments(algo_name, dataset_name, varargin)
close all;
diary off;
addpath('graph');
addpath(genpath('data'));
if exist(fullfile(pwd, 'spams', 'build'), 'dir')
    addpath(fullfile(pwd, 'spams', 'build'));
end

if nargin == 0
    algo_name = 'KMEANS';
    dataset_name = 'glass';
end
params = consts(algo_name, dataset_name, varargin{:});


%% setup logging and snapshot path
time_stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
if ~isempty(params.log_path)
    if ~exist(params.log_path, 'dir')
        mkdir(params.log_path);
    end
    diary(fullfile(params.log_path, sprintf('%s-%s-%s.log', algo_name, dataset_name, time_stamp)));
end
if ~isempty(params.cache_path) && ~exist(params.cache_path, 'dir')
    mkdir(params.cache_path);
end


%% parameters
disp('Parameters:');
disp(params);
fprintf('\n\n');
num_runs = params.num_runs;
iterations = params.iterations;
verbose = params.verbose;
parallel = params.parallel;
doplot = params.plot;
permdata = params.permdata;
cache_path = params.cache_path;
if params.seed > 0
    rng(params.seed);
else
    rng('shuffle');
end
params = rmfield(params, 'seed');
params = rmfield(params, 'plot');
params = rmfield(params, 'log_path');
params = rmfield(params, 'cache_path');
params = rmfield(params, 'permdata');


%% load data
[fea, gnd] = load_data(dataset_name);
X = fea';
size(gnd)
Y = to_categorical(gnd);
fprintf('Dataset statistics of %s:\n', dataset_name);
fprintf('num_clusters: %d\n', size(Y, 1));
fprintf('num_features: %d\n', size(X, 1));
fprintf('num_samples:  %d\n\n', size(X, 2));
if permdata
    perm = randperm(size(X, 2));
    X = X(:, perm);
    Y = Y(:, perm);
end

% normalization
X = normalize(X', 'range', [-1,1]); % same as mapminmax but to columns
X = X';


%% function handles
func_handle = eval(sprintf('@run%s', algo_name));

%% Do actual training
% nun_runs x param_sets x 3 (ACC/NMI/Purity)
tt = tic;
[results_mat, param_grid] = meta_runner(func_handle, X, Y, params);
fprintf('Done %s on %s in %.2fs.\n', algo_name, dataset_name, toc(tt));

fprintf('\n\n=================== Final Results of %s on %s =====================\n\n', algo_name, dataset_name);
acc_record = zeros(numel(param_grid), 7);
std_record = zeros(numel(param_grid), 7);
for i = 1:numel(param_grid)
    results_i = results_mat(:, i, :);
    mean_i = squeeze(mean(results_i, 1));
    std_i = squeeze(std(results_i, 0, 1));
    fprintf('Parameters %d/%d:\n', i, numel(param_grid));
    fprintf('ACC: %.4f (+-%.4f)\n', mean_i(1), std_i(1));
    fprintf('NMI: %.4f (+-%.4f)\n', mean_i(2), std_i(2));
    fprintf('PUR: %.4f (+-%.4f)\n', mean_i(3), std_i(3));
    fprintf('PRE: %.4f (+-%.4f)\n', mean_i(5), std_i(5));
    fprintf('REC: %.4f (+-%.4f)\n', mean_i(6), std_i(6));
    fprintf('F: %.4f (+-%.4f)\n', mean_i(4), std_i(4));
    fprintf('ARI: %.4f (+-%.4f)\n', mean_i(7), std_i(7));
    disp(param_grid(i));
    acc_record(i, :) = mean_i;
    std_record(i, :) = std_i;
end
[results_highest, selected] = max(acc_record, [], 1);
% results_std = [std_record(selected(1), 1), std_record(selected(2), 2), std_record(selected(3), 3)];
results_std = [std_record(selected(1), 1), std_record(selected(2), 2), std_record(selected(3), 3), std_record(selected(5), 5), std_record(selected(6), 6), std_record(selected(4), 4), std_record(selected(7), 7)];
fprintf('Highest accuracy for %s on %s over %d runs among %d parameter sets: %.4f(+-%.4f), parameters:\n', ...
    algo_name, dataset_name, num_runs, numel(param_grid), results_highest(1), results_std(1));
best_params_acc = param_grid(selected(1));
disp(best_params_acc);

fprintf('Highest NMI for %s on %s over %d runs among %d parameter sets: %.4f(+-%.4f), parameters:\n', ...
    algo_name, dataset_name, num_runs, numel(param_grid), results_highest(2), results_std(2));
best_params_nmi = param_grid(selected(2));
disp(best_params_nmi);

fprintf('Highest purity for %s on %s over %d runs among %d parameter sets: %.4f(+-%.4f), parameters:\n', ...
    algo_name, dataset_name, num_runs, numel(param_grid), results_highest(3), results_std(3));
best_params_purity = param_grid(selected(3));
disp(best_params_purity);

fprintf('Highest precision for %s on %s over %d runs among %d parameter sets: %.4f(+-%.4f), parameters:\n', ...
    algo_name, dataset_name, num_runs, numel(param_grid), results_highest(5), results_std(5));
best_params_precision = param_grid(selected(5));
disp(best_params_precision);

fprintf('Highest recall for %s on %s over %d runs among %d parameter sets: %.4f(+-%.4f), parameters:\n', ...
    algo_name, dataset_name, num_runs, numel(param_grid), results_highest(6), results_std(6));
best_params_recall = param_grid(selected(6));
disp(best_params_recall);

fprintf('Highest fscore for %s on %s over %d runs among %d parameter sets: %.4f(+-%.4f), parameters:\n', ...
    algo_name, dataset_name, num_runs, numel(param_grid), results_highest(4), results_std(4));
best_params_fscore = param_grid(selected(4));
disp(best_params_fscore);

fprintf('Highest ARI for %s on %s over %d runs among %d parameter sets: %.4f(+-%.4f), parameters:\n', ...
    algo_name, dataset_name, num_runs, numel(param_grid), results_highest(7), results_std(7));
best_params_ARI = param_grid(selected(7));
disp(best_params_ARI);

fprintf('for recording:\n%.4f(+-%.4f)\n%.4f(+-%.4f)\n%.4f(+-%.4f)\n%.4f(+-%.4f)\n%.4f(+-%.4f)\n%.4f(+-%.4f)\n%.4f(+-%.4f)\n', ...
    results_highest(1), results_std(1), ...
    results_highest(2), results_std(2), ...
    results_highest(3), results_std(3), ...
    results_highest(5), results_std(5), ...
    results_highest(6), results_std(6), ...
    results_highest(4), results_std(4), ...
    results_highest(7), results_std(7));

if exist(cache_path, 'dir')
    save(fullfile(cache_path, sprintf('results-%s-%s-%s.mat', algo_name, dataset_name, time_stamp)), 'results_mat', 'param_grid');
    save(fullfile(cache_path, sprintf('best_params-%s-%s-%s.mat', algo_name, dataset_name, time_stamp)), ...
        'best_params_acc', 'best_params_nmi', 'best_params_purity', 'results_highest', 'results_std');
end
fprintf('\n\n=================== Final Results of %s on %s =====================\n\n', algo_name, dataset_name);

%% optionally plot accuracy results
if length(doplot) == 2
    plot_acc_bar(param_grid, results_mat(:, :, 1), doplot{1}, doplot{2});
end
%% post-processing
diary off;

%% return values
if nargout > 0
    varargout{1} = results_highest;
end
if nargout > 1
    varargout{2} = [std_record(selected(1), 1), std_record(selected(2), 2), std_record(selected(3), 3)];
end
if nargout > 2
    varargout{3} = best_params_acc;
end


end

%% function handles


%% ------------------------------------------------------------------------
function labels = runKMEANS(X, Y, params)
% if params.verbose
%     do_disp = 'final';
% else
%     do_disp = 'off';
% end
% nCls = length(unique(Y));
nCls = size(Y, 1);
fprintf('nCls:%d\n', nCls);
% whether should run kmeans in parallel
% labels = kmeans(X', size(Y, 1), 'Replicates', 100, 'MaxIter', params.iterations, ...
%     'Start', 'plus', 'Display', do_disp);

[labels, ~] = litekmeans(X', nCls, 'MaxIter', 100);

end

%% ------------------------------------------------------------------------
function labels = runUFSELM(X, Y, params)

nNbr = getfield_with_default(params, 'nNbr', 10);

selftune = getfield_with_default(params, 'selftune', false);

if selftune
    lapla_norm = getfield_with_default(params, 'lapla_norm', true);
    [A, ~] = selftuning(X', nNbr);
    L = Adjacency2Laplacian(A, lapla_norm);
else
    options.NN = nNbr;
    options.GraphDistanceFunction = getfield_with_default(params, ...
        'GDF', 'euclidean');
% options.GraphDistanceFunction: 'euclidean' | 'cosine'
    options.GraphWeights = getfield_with_default(params, 'GW', 'distance');
% options.GraphWeights: 'distance' | 'heat'
    options.LaplacianNormalize=1;
    options.LaplacianDegree=1;
    options.GraphWeightParam = 1;
    L = laplacian(options, X');
end

nNrn = getfield_with_default(params, 'nNrn', 1024);
var1 = getfield_with_default(params, 'var1', 6);
var1 = 10^var1;
var2 = getfield_with_default(params, 'var2', -6);
var2 = 10^var2;
% nCls = length(unique(Y));
nCls = size(Y, 1);
[~, F] = UFSELM(X', L, nCls, var1, var2, nNrn);
[~, labels] = max(F,[],2);
end

%% ------------------------------------------------------------------------
function labels = runSC(X, Y, params)
% Run spectral clustering.
% number of nearest neighbors
NN = getfield_with_default(params, 'NN', 15);
normalize = getfield_with_default(params, 'normalize', true);
selftune = getfield_with_default(params, 'selftune', false);
output_dim = getfield_with_default(params, 'output_dim', size(Y, 1));
if selftune
    [W, ~] = selftuning(X', NN);
else
    weights = getfield_with_default(params, 'weights', 'heat');
    scale = getfield_with_default(params, 'scale', 0);
    options = struct('NN', NN, 'GraphDistanceFunction', 'euclidean', ...
        'GraphWeights', weights, 'GraphWeightParam', scale);
    W = adjacency(options, X');
end
labels = spectral_clustering(W, size(Y, 1), normalize, output_dim);

end

%% ------------------------------------------------------------------------
function labels = runUSELM(X, Y, params)
NN = getfield_with_default(params, 'nNbr', 5);
scale = getfield_with_default(params, 'scale', 0);
lambda = getfield_with_default(params, 'lambda', 0); % -4:1:4
lambda = 10^lambda;
hidden_dim = getfield_with_default(params, 'hidden_dim', 1024);
output_dim = getfield_with_default(params, 'output_dim', 32);
normalize = getfield_with_default(params, 'normalize', true);
selftune = getfield_with_default(params, 'selftune', false);

if selftune
    lapla_norm = getfield_with_default(params, 'lapla_norm', true);
    [A, ~] = selftuning(X', NN);
    L = Adjacency2Laplacian(A, lapla_norm);
else
    options.NN = NN;
    options.GraphDistanceFunction = getfield_with_default(params, ...
        'GDF', 'euclidean');
% options.GraphDistanceFunction: 'euclidean' | 'cosine' | 'hamming_distance'
    options.GraphWeights = getfield_with_default(params, 'GW', 'distance');
% options.GraphWeights: 'distance' | 'binary' | 'heat'
    options.LaplacianNormalize=1;
    options.LaplacianDegree=1;
    options.GraphWeightParam = 1;
    L = laplacian(options, X');
end

elmModel = uselm(X', L, options);
[labels, ~] = litekmeans(elmModel.Embed, nCls, 'MaxIter', 100);

% labels = uselm_interface(X, size(Y, 1), L,  lambda, hidden_dim, output_dim, normalize);
end

%% ------------------------------------------------------------------------
function labels = runELMcLDA(X, Y, params)
paras.K = size(Y, 1);
paras.y = Y;
nNbr = getfield_with_default(params, 'nNbr', 10);
var1 = getfield_with_default(params, 'var1', 6); % -6:1:6
paras.lambda = 10^var1;

% L = laplacian(options, X');
% use selftuning
selftune = getfield_with_default(params, 'selftune', false);
if selftune
    lapla_norm = getfield_with_default(params, 'lapla_norm', true);
    [A, ~] = selftuning(X', nNbr);
    L = Adjacency2Laplacian(A, lapla_norm);
else
    options.NN = getfield_with_default(params, 'nNbr', 10);
    options.GraphDistanceFunction = getfield_with_default(params, ...
        'GDF', 'euclidean');
    options.GraphWeights = getfield_with_default(params, 'GW', 'distance');
% options.GraphDistanceFunction: 'euclidean' | 'cosine' | 'hamming_distance'
% options.GraphWeights: 'distance' | 'binary' | 'heat'
    options.LaplacianNormalize=1;
    options.LaplacianDegree=1;
    options.GraphWeightParam = 1;
    L = laplacian(options, X');
end

paras.NumHiddenNeuron = getfield_with_default(params, 'nNrn', 1024);

[labels, ~, ~]=elmc_lda(X', paras);
end
%% ------------------------------------------------------------------------
function labels = runELMJEC(X, Y, params)

nOut = getfield_with_default(params, 'nOut', 32);
nNbr = getfield_with_default(params, 'nNbr', 10);
nNrn = getfield_with_default(params, 'nNrn', 1024);
var1 = getfield_with_default(params, 'var1', 0);
var1 = 2^var1;
var2 = getfield_with_default(params, 'var2', 0);
var2 = 2^var2;

selftune = getfield_with_default(params, 'selftune', false);
if selftune
    lapla_norm = getfield_with_default(params, 'lapla_norm', true);
    [A, ~] = selftuning(X', nNbr);
    L = Adjacency2Laplacian(A, lapla_norm);
else
    options.NN = nNbr;
    options.GraphDistanceFunction = getfield_with_default(params, ...
        'GDF', 'euclidean');
    options.GraphWeights = getfield_with_default(params, 'GW', 'distance');
% options.GraphDistanceFunction: 'euclidean' | 'cosine' | 'hamming_distance'
% options.GraphWeights: 'distance' | 'binary' | 'heat'
    options.LaplacianNormalize=1;
    options.LaplacianDegree=1;
    options.GraphWeightParam = 1;
    L = laplacian(options, X');
end

% nCls = length(unique(Y));
nCls = size(Y, 1);
[F] = ELMJEC(X', L, nCls, var1, var2, nOut, nNrn);
[~, labels] = max(F,[],2);
end

%% ------------------------------------------------------------------------
function labels = runUFSELML2(X, Y, params)

options.NN = getfield_with_default(params, 'nNbr', 10);

options.GraphDistanceFunction = getfield_with_default(params, ...
    'GDF', 'euclidean');
options.GraphWeights = getfield_with_default(params, 'GW', 'distance');
% options.GraphDistanceFunction: 'euclidean' | 'cosine' | 'hamming_distance'
% options.GraphWeights: 'distance' | 'binary' | 'heat'
options.LaplacianNormalize=1;
options.LaplacianDegree=1;
options.GraphWeightParam = 1;

% L = laplacian(options, X');
% use selftuning
lapla_norm = getfield_with_default(params, 'lapla_norm', true);
[A, ~] = selftuning(X', options.NN);
L = Adjacency2Laplacian(A, lapla_norm);

nNrn = getfield_with_default(params, 'nNrn', 1024);
var1 = getfield_with_default(params, 'var1', 6);
var1 = 10^var1;
var2 = getfield_with_default(params, 'var2', -6);
var2 = 10^var2;
% nCls = length(unique(Y));
nCls = size(Y, 1);
[~, F] = UFSELM(X', L, nCls, var1, var2, nNrn);
[~, labels] = max(F,[],2);
end
