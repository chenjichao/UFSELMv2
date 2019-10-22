function [SS, varargout] = ASC_interface(X, C, eta, gamma, NN, iterations, impl)
%% Interface to dispatch to existing implementations.
if nargin < 7
    impl = 'RLLC';
end
switch lower(impl)
    case 'rllc'
        [SS, flag] = ASC_admm_rllc(X, C, eta, gamma, NN, iterations);
    case 'llc'
        [SS, flag] = ASC_llc2(X, C, eta, gamma, NN, iterations);
    case 'local'
        [SS, flag] = ASC_admm_l2_local(X, C, eta, gamma, NN, iterations);
    otherwise
        error('Unknown implementation method: %s', impl);
end
if nargout > 1
    varargout{1} = flag;
end
end