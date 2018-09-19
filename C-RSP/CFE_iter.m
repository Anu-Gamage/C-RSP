function [varargout] = CFE_iter(A, n, k, m, b, labels)
% Common - Free Energy
% Inputs:
% A - multilayer affinity tensor
% n - no. of nodes
% k - no. of clusters
% m - no. of layers
% b - FE tuning paramter
% labels - true cluster assignments of nodes
% Outputs:
% If only one, wanted distances, so return dRSP
% else, return metrics:
%   acc_arr - accuracy 
%   nmi_arr - normalized mutual information
%   final_labels - estimated labels from RSP
% Anuththari Gamage, 3/22/2018
% Modified Brian Rappaport, 7/4/2018
% Modified for FE, Anuththari Gamagae 9/7/2018
% Modified for CFE_iter Anuththari Gamage, 9/28/2018 

    infFlag = 1e12;
    n = size(A{1},1);
    P_ref = cell(1,m);                  % Reference transition probability
    for i = 1:m
        node_degrees = sum(A{i},2);
        node_degrees(node_degrees == 0) = infFlag;
        inv_D = sparse(1:n, 1:n, 1./node_degrees);  % Inverse of Degree matrix
        P_ref{i} = inv_D*A{i};      
    end
       
    % Construct common W
    C = cellfun(@(A) 1./(A + infFlag*(A==0)),A,'un',0);           % Convert A into C
    C_joint = combine_C(C,infFlag);            % Combined cost matrix   
    P_joint = combine_P(P_ref);                % Combines probability matrix
    W = P_joint.*exp(-b*C_joint);              % Combined weights
   
    specRadius = eigs(W,1);                    % Convergence check
    if specRadius >= 1
     %   error('Will not converge')
        disp('Will not converge')
    end
    
    dFE = fe_iterative(P_joint,C_joint, b);
    
    if nargout == 1
        varargout{1} = dFE;
    else
        % Spectral Clustering   
        aff = 1./(eye(n) + dFE) - eye(n);      % Affinity Matrix
        D = diag(1./sqrt(sum(aff,2)));
        L = D*aff*D;          % Normalized Laplacian
        [V,~] = eigs(L,k+1);
        V = V./sqrt(sum(V.^2,2));
        V = V(:,2:end);

        [final_labels,acc_arr,nmi_arr] = postproc(V,k,labels);
        varargout{1} = acc_arr;
        varargout{2} = nmi_arr;
        varargout{3} = final_labels;
    end
end

function new_C = combine_C(C,infFlag)
    m = numel(C);
    new_C = C{1}.*(C{1} < infFlag/2);
    nz_C = C{1}~=0;         % Tracks count of non-zero costs
    for layers = 2:m
        new_C = new_C + C{layers}.*(C{layers} < infFlag/2);
        nz_C = nz_C + (C{layers}~=0);
    end
    new_C = new_C./(nz_C + (nz_C==0));
end


function new_P = combine_P(P)
    n = size(P{1},1);
    m = numel(P);
    mask = false(n,n,m);
    new_P = ones(n);
    for ii = 1:m
        mask(:,:,ii) = P{ii} ~= 0;
        new_P(mask(:,:,ii)) = new_P(mask(:,:,ii)).*P{ii}(mask(:,:,ii));
    end
    roots = sum(mask,3);
    new_P = nthroot(new_P,roots);
    new_P(isnan(new_P)) = 0;
    new_P = new_P./(sum(new_P,2));     % Make row stochastic
    new_P(isnan(new_P)) = 0;
end

function dFE = fe_iterative(P,C,beta)
% working on the assumption that succ(i) = nodes j with p_ij > 0 (not based
% on original adjacency graphs)
% theta = beta

n = size(P,1);
%dFE = rand(n);
%dFE(1:n+1:end) = 0; % clear diagonal

Z = exp(-beta*rand(n));
Z(1:n+1:end) = 1;
for ii = 1:floor(n/2)
    selection_set = rand(1,n) > 0.7;
    for nbr = find(selection_set)
        % find neighbors
        succ = find(P(nbr,:) > 1e-3);
        Z(nbr,:) = sum(P(nbr,succ).*exp(-beta*(C(nbr,succ))).*Z(succ,:)',2);
        Z(nbr,nbr) = 1;
%         dFE(nbr,:) = -1/beta*log(sum(P(nbr,succ).*exp(-beta*(C(nbr,succ) + dFE(succ,:)')),2));
%         dFE(nbr,nbr) = 0;
    end
end
dFE = -1/beta*log(Z);
end