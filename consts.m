function params = consts(algo_name, dataset_name, varargin)
% Best params for each algorithm and each dataset.
param_func = sprintf('%s4%s', algo_name, dataset_name);
if exist(param_func) ~= 0
    params = eval(param_func);
else
    params = [];
end

ip = inputParser;
ip.KeepUnmatched = true;
ip.PartialMatching = false;
ip.addParameter('verbose', false, @islogical);
ip.addParameter('iterations', 50, @isnumeric);
ip.addParameter('num_runs', 10, @isnumeric);
ip.addParameter('parallel', true, @islogical);
ip.addParameter('seed', -1, @isnumeric);
ip.addParameter('plot', {}, @iscell);
ip.addParameter('log_path', 'logs', @ischar);
ip.addParameter('cache_path', 'cache', @ischar);
ip.addParameter('fail_on_error', true, @islogical);
ip.addParameter('permdata', true, @islogical);
ip.parse(varargin{:});
names1 = fieldnames(ip.Results);
names2 = fieldnames(ip.Unmatched);
for i = 1:numel(names1)
    params.(names1{i}) = ip.Results.(names1{i});
end
for i = 1:numel(names2)
    params.(names2{i}) = ip.Unmatched.(names2{i});
end

end