function [results_mat, varargout] = meta_runner(func_handle, X, Y, params)
% Default entry for running different algorithms.
% The FUNC_HANDLE should have the following signature:
% LABELS = FUNC_HANDLE(X, Y, PARAMS)


num_runs = params.num_runs;
parallel = params.parallel;
fail_on_error = params.fail_on_error;
params = rmfield(params, 'num_runs');
params = rmfield(params, 'fail_on_error');

time_grid = [];
param_grid = generate_tunable_param_grid(params);
accuracies = cell(num_runs*numel(param_grid), 1);
total_runs = num_runs * numel(param_grid);
if parallel
    num_workers = min(feature('numcores'), total_runs);
else
    num_workers = 0;
end
[~, gt] = max(Y, [], 1);
parfor (ii = 1:total_runs, num_workers)
    tii = tic;
    [r, c] = ind2sub([num_runs, numel(param_grid)], ii);
    if fail_on_error
        labels = func_handle(X, Y, param_grid(c));
    else
        try
            labels = func_handle(X, Y, param_grid(c));
        catch ME
            warning('Error appeared at %d: %s', ii, ME.message);
            labels = [];
        end
    end

    if isempty(labels)
        result = [0 0 0 0 0 0 0];
    else
%         result = ClusteringMeasure(gt, labels);
        result(1:3) = ClusteringMeasure(gt, labels);
        result(4:6) = compute_f(gt, labels');
        result(7) = RandIndex(gt, labels);
    end
    temptime = toc(tii);
    accuracies{ii} = [result, temptime];
%     fprintf('Done experiment %d/%d in %.2fs, accuracy: %.4f, nmi: %.4f, purity: %.4f, parameter:\n',...
%         ii, total_runs, toc(tii), result(1), result(2), result(3));
    fprintf('Done experiment %d/%d in %.2fs, accuracy: %.4f, nmi: %.4f, purity: %.4f, precision: %.4f, recall: %.4f, F-score: %.4f, ARI: %.4f, parameter:\n',...
        ii, total_runs, temptime, result(1), result(2), result(3), result(5), result(5), result(4), result(7));
    disp(param_grid(c));
end
results_mat = zeros(num_runs, numel(param_grid), 8);
for ii = 1:total_runs
    [r, c] = ind2sub([num_runs, numel(param_grid)], ii);
    results_mat(r, c, 1:length(accuracies{ii})) = accuracies{ii};
end
if nargout > 1
    varargout{1} = param_grid;
end
end

%% utilities for generating proper grid
function param_grid = generate_tunable_param_grid(params)
if isempty(params)
    param_grid.empty_placeholder = 1;
    return;
end
if isfield(params, 'inner_loop')
    inner_loop = params.inner_loop;
    params = rmfield(params, 'inner_loop');
else
    inner_loop = [];
end
if isfield(params, 'tunable')
    param_grid = generate_parameter_grid(params.tunable);
    fixed_params = rmfield(params, 'tunable');
    names = fieldnames(fixed_params);
    for i = 1:numel(names)
        for j = 1:numel(param_grid)
            param_grid(j).(names{i}) = fixed_params.(names{i});
        end
    end
else
    param_grid = generate_parameter_grid(params);
end
if ~isempty(inner_loop)
    names = fieldnames(inner_loop);
    for i = 1:numel(names)
        for j = 1:numel(param_grid)
            param_grid(j).(names{i}) = inner_loop.(names{i});
        end
    end
end
end