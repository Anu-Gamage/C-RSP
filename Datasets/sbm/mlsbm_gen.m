function [A, labels] = mlsbm_gen(n,k,m,cin,lambda)
% Generates multi-layers SBM graphs
% Inputs:
% n - number of nodes
% k - number of communities
% m - number of layers
% cin - intra-cluster node degree
% lambda - intra-cluster node degree
% Outputs: 
% A - cell array of SBM graphs
% labels - cluster assignment for each node
% Anuththari Gamage, 3/16/2018 

    A = cell(1,m);
    cout = (1-lambda)*cin;
    
    tm = clock;                         % Shuffle random numbers
    rng(round(tm(6)), 'twister');
    
    conn = 1:n;
    for layers = 1:m
        seed = randi(1000);
        [G,labels] = sbm_gen(n,k,cin,cout,seed);

        A{layers} = G;
         % Check for isolated nodes and fix
        conn = intersect(find(any(G,2)),conn);
    end
    labels = labels(conn);
    for layers = 1:m
        A{layers} = A{layers}(conn,conn);
%         if ~isempty(isol)
%             for idx = 1:numel(isol)
%                neighbors = find(labels == labels(isol(idx)));
%                edge = datasample(neighbors, 1);
%                G(isol(idx),edge) = 1;
%                G(edge, isol(idx)) = 1;
%             end
%         end
    end
end