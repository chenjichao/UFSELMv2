function eigvector = eig_decom(ddata,ReducedDim)
% option.tol = 1e-9;
option.issym = 1;
%     option.disp=0;
option = struct('disp',0);
% option.maxit = 100;

ddata = max(ddata, ddata');
%     ddata = (ddata+ddata')/2;
%     ddata = ddata +1e-10*eye(size(ddata,1));
dimMatrix = size(ddata,2);

if dimMatrix > 2000 && ReducedDim < dimMatrix/10  % using eigs to speed up!
    [eigvector, eigvalue] = eigs(ddata,ReducedDim,'la',option);
    eigvalue = diag(eigvalue);
else
    [eigvector, eigvalue] = eig(ddata);
    eigvalue = diag(eigvalue);
    
    [~, index] = sort(-eigvalue);
    eigvalue = eigvalue(index);
    eigvector = eigvector(:, index);
end

clear ddata;
% maxEigValue = max(abs(eigvalue));
%     eigIdx = find(abs(eigvalue)/maxEigValue < 1e-12);
%     eigvalue (eigIdx) = [];
%     eigvector (:,eigIdx) = [];

if ReducedDim < length(eigvalue)
    eigvalue = eigvalue(1:ReducedDim);
    eigvector = eigvector(:, 1:ReducedDim);
end

end