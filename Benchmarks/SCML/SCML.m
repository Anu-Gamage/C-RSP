function [varargout] = SCML(A, k, lambda_scml, labels)
% Modified function file for SC-ML
% Inputs:
% A - multilayer graph tensor (full, not sparse)
% labels - true cluster assignments of nodes
% k - no. of clusters
% Outputs:
% If only one, wanted just embeddings, so return emb
% else, return metrics:
%   acc_arr - accuracy 
%   nmi_arr - normalized mutual information
%   final_labels - estimated labels from SC-ML
% Anuththari Gamage, 3/19/2018
% Modified Brian Rappaport, 7/4/2018

        % Convert to dense matrix
        n = size(A{1},1);
        m = numel(A);
        G = zeros(n,n,m);
        for i = 1:m
            G(:,:,i) = full(A{i});
        end
        A = G;
        
        % Run SC-ML
        emb = sc_ml(A,k,lambda_scml); 
        
        if nargout == 1
            varargout{1} = emb;
        else
            [final_labels,acc_arr,nmi_arr] = postproc(emb,k,labels);
            varargout{1} = acc_arr;
            varargout{2} = nmi_arr;
            varargout{3} = final_labels;
        end
end