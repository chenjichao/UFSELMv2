function [features, varargout] = extract_random_features(data,dims)
rand_matrix = randn(dims, size(data, 1));
rand_matrix = bsxfun(@rdivide, rand_matrix, sqrt(sum(rand_matrix.^2, 2)+eps));
features = rand_matrix * data;
if nargout > 1
    varargout{1} = rand_matrix;
end
end

