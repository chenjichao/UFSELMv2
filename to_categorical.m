function cate = to_categorical(labels, negflag)
% Convert labels to one-hot encoding.
% If negflag is not specified, then it is set as 0.
num_classes = numel(unique(labels));
N = numel(labels);
cate = full(sparse(labels(:)', 1:N, ones(1, N), num_classes, N));
if nargin > 1
    cate(cate==0) = negflag;
end

end