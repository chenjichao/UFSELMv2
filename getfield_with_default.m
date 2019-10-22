function [value, varargout] = getfield_with_default(params, name, value)
if isfield(params, name)
    value = getfield(params, name);
else
    params.(name) = value;
end
% return (maybe) augmented parameter struct
if nargout > 1
    varargout{1} = params;
end
end