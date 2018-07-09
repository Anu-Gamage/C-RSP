function [acc_arr, nmi_arr, final_labels] = CFE(A, C, labels, n, k, m, b)
% Common - Free Energy Distance
% Inputs:
% A - multilayer graph tensor
% C - cost tensor
% labels - true cluster assignments of nodes
% n - no. of nodes
% k - no. of clusters
% m - no. of layers
% b - FE tuning paramter
% Outputs:
% acc_arr - accuracy 
% nmi_arr - normalized mutual information
% final_labels - estimated labels from RSP
% Anuththari Gamage, 3/24/2018

    P_ref = cell(1,m);                  % Reference transition probability
    for i = 1:m
        node_degrees = sum(A{i},2);
        inv_D = sparse(1:n, 1:n, 1./node_degrees);  % Inverse of Degree matrix
        P_ref{i} = inv_D*A{i};      
    end
       
    % Construct common W
    C_joint = combine_C(C);                  % Combined cost matrix   
    P_joint = combine_P(P_ref);                % Combines probability matrix
    W = P_joint.*exp(-b*C_joint);              % Combined weights
    
    specRadius = eigs(W,1);                    % Convergence check
    if specRadius >= 1
        error('Will not converge')
    end
    
    Z = inv(speye(n) - W);
    Dh = diag(diag(Z));
    Zh = Z*inv(Dh);
    phi = (-1/b)*log(Zh);
    dFE = (phi + phi')./2;   
    infFlag = 1e12;
    dFE(dFE == inf) = infFlag;                 % Flagging inf
    
    % Spectral Clustering   
    aff = 1./(eye(n) + dFE) - eye(n);      % Affinity Matrix
    D = diag(sum(aff,2)) ;
    L = (D^(-1/2))*aff*(D^(-1/2));          % Normalized Laplacian
    [V,E] = eig(L);
    [~,I] = sort(diag(E),'descend');
    V = V(:, I(2:k+1)');                    % Changed to take from second largest ei.value onwards
    V = V./sqrt(sum(V.^2,2));
    
    % Clustering using k-Means and Linear Sum Assignment
    est_labels = kmeans(V, k)';
    final_labels = zeros(1, n);

    C = confusionmat(labels, est_labels);
    new_labels = munkres(-C);
    if ~isequal(new_labels, 1:length(new_labels))
       for i = 1:length(new_labels)
           final_labels(est_labels == new_labels(i)) = i;
       end
    else
        final_labels = est_labels;
    end
    % Accuracy
    acc_arr = 100*sum(labels == final_labels)/n;
    nmi_arr = nmi(labels, final_labels);
end

function new_C = combine_C(C)
    m = size(C,2); 
    new_C = C{1};
    nz_C = C{1}~=0;         % Tracks count of non-zero costs
    for layers = 2:m
        new_C = new_C + C{layers};
        nz_C = nz_C + (C{layers}~=0);
    end
    new_C = new_C./(nz_C + (nz_C==0));
end

function new_P = combine_P(P)
   m = size(P,2);              
   new_P = matrix_mult(P);
   
   % Taking nth roots
   roots = (P{1}~=0);
   for i = 2:m
       roots = roots + (P{i}~=0);
   end
   roots = roots + (roots==0);          % To eliminate 0th roots
   new_P = nthroot(new_P, roots);
   new_P = new_P./(sum(new_P,2));     % Make row stochastic
end

function new_P = matrix_mult(P)
   m = size(P,2); 
   % Multiplication retaining all edges
   no_edges = (P{1} == 0);              % Record entries with no edges
   for i = 2:m
       no_edges =  no_edges.*(P{i}==0);
   end    
   
   Pnz = cell(1,m);                 % Add 1 to avoid multiplication error
   for i = 1:m
       Pnz{i} = P{i} + (P{i}==0);    
   end
   new_P = Pnz{1};
   for i = 2:m
    new_P = new_P.*Pnz{i};
   end
   new_P = new_P.*(no_edges ~= 1);       % Zero out entries with no edges 
end


