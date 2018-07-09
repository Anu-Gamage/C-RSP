function [acc_val, nmi_val] = GMultiNMF(X, K, W, label,options)
%function [finalU, finalV, finalcentroidV, log] = GMultiNMF(X, K, W, label,options)
%	Notation:
% 	X ... a cell containing all views for the data
% 	K ... number of hidden factors
% 	W ... weight matrix of the affinity graph 
% 	label ... ground truth labels

%	Writen by Jialu Liu (jliu64@illinois.edu)
% 	Modified by Zhenfan Wang (zfwang@mail.dlut.edu.cn)

%	References:
% 	J. Liu, C.Wang, J. Gao, and J. Han, ??Multi-view clustering via joint nonnegative matrix factorization,?? in Proc. SDM, Austin, Texas, May 2013, vol. 13, pp. 252?C260.
% 	Zhenfan Wang, Xiangwei Kong, Haiyan Fu, Ming Li, Yujia Zhang, FEATURE EXTRACTION VIA MULTI-VIEW NON-NEGATIVE MATRIX FACTORIZATION WITH LOCAL GRAPH REGULARIZATION, ICIP 2015.

%Note that columns are data vectors here

    viewNum = length(X);
    rounds = options.rounds;
    nSmp = size(X{1},2);                          %Number of data points

    U = cell(1,viewNum);
    V = cell(1,viewNum);

    % initialize basis and coefficient matrices, initialize on the basis of
    % standard GNMF algorithm

    Goptions.alpha=options.Gaplpha;
    
    [U{1},V{1}] = GNMF(X{1},K,W{1},options);
    for i = 2:viewNum
        [U{i}, V{i}] = GNMF(X{i},K,W{i},Goptions);
    end

    % Alternate Optimisations for consensus matrix and individual view matrices
    options.alphas = options.alphas/sum(options.alphas);
    % log = zeros(1,rounds);
    maxac = 0;                              %Maximum accuracy
    for j = 1:rounds                        %Number of rounds of AO
        if j == 1
            centroidV = V{1};                       %Basic initialization for consensus matrix
        else
            centroidV = reshape(reshape(cell2mat(V),numel(V{1}),[])*options.alphas',size(V{1}));
        end
    %     logL = 0;                                   %Loss for the round
    %     for i = 1:viewNum
    %         alpha = options.alphas(i);
    %         if alpha > 0
    %             Wtemp = options.beta*alpha*W{i};            %Modify the weight matrix with the involved parameters
    %             DCol = full(sum(Wtemp,2));
    %             D = spdiags(DCol,0,nSmp,nSmp);
    %             L = D - Wtemp;                              %Get matrix L
    %             if isfield(options,'NormW') && options.NormW
    %                 D_mhalf = spdiags(DCol.^-.5,0,nSmp,nSmp) ;
    %                 L = D_mhalf*L*D_mhalf;
    %             end
    %         else
    %             L = [];
    %         end
    %         % Compute the losses
    %         tmp1 = X{i} - U{i}*V{i}';
    %         tmp2 = V{i} - centroidV;
    %         tmp3 = L'*V{i}.*V{i};
    %         logL = logL + sum(tmp1(:).^2) + alpha*sum(tmp2(:).^2) + sum(tmp3(:));  %????????SampleW??V'*L*V
    %     end
    %     log(j) = logL;

        [acc_val,nmi_val] = printResult(centroidV,label,options.K,options.kmeans);
        acc_val = 100*acc_val;
        if acc_val > maxac
            maxac = acc_val;
            finalU = U;
            finalV = V;
    %         finalcentroidV = centroidV;
        end

        for i = 1:viewNum
            options.alpha = options.alphas(i);
            [U{i},V{i}] = PerViewNMF(X{i},K,centroidV,W{i},options,finalU{i},finalV{i});
            %Peform optimization with V* (centroidV) fixed and inits finalU, finalV
        end
    end
end