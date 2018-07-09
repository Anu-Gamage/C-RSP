% Main test file for C-RSP vs benchmarks
% Anuththari Gamage, Brian Rappaport
% 6/29/2018

saveoutput = 1;

tic
% Algorithms used for comparison: 1 = C-RSP, 2 = SC-ML, 3 = CoregSC
alg_names = {'CRSP', 'SCML', 'CSC'};
algs = 1:3;
% Datasets used: 1 = UCI, 2 = 3Sources, 3 = MultiviewTwitter
dataset_names = {'uci','three','mvt'};
datasets = 1;

fp = 1;

results = struct;
num_reps = 10;
for jj = datasets
    fprintf(fp,'-----------------------\n');
    switch dataset_names{jj}
        case 'uci'
            if ~exist('uci','var')
                uci = readuci();
            end
            
            A = struct2cell(uci.A);
            labels = uci.labels;
            fprintf(fp,'UCI:\n');
        case 'three'
            load('Datasets/3sources/sources_data.mat');
            
            A = sources_data{1};
            labels = sources_data{3};
%         %    if ~exist('three','var')
%                 three = read3s();
%         %    end
%             
%             A = struct2cell(three.A);
%             labels = three.labels;
            fprintf(fp,'3Sources:\n');
        case 'mvt'
            if ~exist('mvt','var')
                mvt = readmvt();
            end
            dataset_choice = mvt.politicsuk;
            
            A = struct2cell(dataset_choice.A);
            labels = dataset_choice.labels;
            fprintf(fp,'Multiview Twitter:\n');
%         case 'sbm'
%             n = 500;
%             k = 2;
%             m = 3;
%             cin = .02*n;
%             lambda = 0.9;
%             [A,labels] = mlsbm_gen(n,k,m,cin,lambda);
%             fprintf(fp,'SBM Data (n = %d, k = %d, m = %d):\n',n,k,m);
        otherwise
    end
    fprintf(fp,'-----------------------\n');
    n = size(A{1},1);
    m = numel(A);
    k = max(labels);
    if size(labels,1) ~= 1
        labels = labels';
    end
    set = cell(1,numel(algs));
    for ii = 1:numel(algs)
        acc_val = zeros(1,num_reps);
        nmi_val = zeros(1,num_reps);
        
        for kk = 1:num_reps
            switch alg_names{algs(ii)}
                case 'CRSP'
                    b = 0.02;                       % inverse temperature parameter
                    [acc_val(kk),nmi_val(kk),~] = CRSP(A,n,k,m,b,labels);
                    fprintf(fp,'C-RSP run %d:\nCCR: %.2f\nNMI: %.2f\n\n',kk,acc_val(kk),nmi_val(kk));
                case 'SCML'
                    lambda_scml = 0.5;              % regularization parameter for SC-ML
                    [acc_val(kk),nmi_val(kk),~] = SCML(A,k,lambda_scml,labels);
                    fprintf(fp,'SC-ML run %d:\nCCR: %.2f\nNMI: %.2f\n\n',kk,acc_val(kk),nmi_val(kk));
                case 'CSC'
                    lambda_coreg = 0.05;            % Co-regularization parameter for PCSC/CCSC
                    num_iter = 1;                   % no. of iterations for PCSC
                    sigma = zeros(1,m);
%                     [acc_val,nmi_val,final_labels] = spectral_pairwise_multiview(A,m,k,sigma,lambda_coreg,labels,num_iter);
                    [acc_val(kk),nmi_val(kk),~] = spectral_centroid_multiview(A,m,k,sigma,repmat(lambda_coreg,1,m),labels',num_iter);
                    fprintf(fp,'Coregularized SC run %d:\nCCR: %.2f\nNMI: %.2f\n\n',kk,acc_val(kk),nmi_val(kk));
%                 case 'MultiNMF'
%                     options = [];                   % Default values below
%                     options.maxIter = 100;          % 100
%                     options.error = 1e-6;           % 1e-6
%                     options.nRepeat = 1;            % 30
%                     options.minIter = 50;           % 50
%                     options.meanFitRatio = 0.1;     % 0.1
%                     options.rounds = 20;            % 20
%                     options.WeightMode='Binary';
%                     options.varWeight = 1;          % 1
%                     options.kmeans = 1;             % 1
%                     options.Gaplpha= 1;             % 1     graph regularization parameter
%                     options.alpha=0.1;              % 0.1
%                     options.delta = 0.1;            % 0.1
%                     options.beta = 10;              % 10
%                     options.gamma = 2;              % 2
%                     options.K = k;
%                     options.alphas = ones(1,m);     % Equal weights to each layer
%                     
%                     [acc_val(kk),nmi_val(kk)] = GMultiNMF(A,k,A,labels',options);
%                     fprintf(fp,'MultiNMF run %d:\nCCR: %.2f\nNMI: %.2f\n\n',kk,acc_val(kk),nmi_val(kk));
                otherwise
            end
%             if strcmp(alg_names{algs(ii)},'MultiNMF')
%                 acc_val = acc_val(kk);
%                 nmi_val = nmi_val(kk);
%                 break; % unnecessary to repeat MultiNMF more than once
%             end
        end
        set{ii} = struct('ccr',acc_val,'nmi',nmi_val);
    end
    results.(dataset_names{jj}) = cell2struct(set,alg_names(algs),2);
end
if saveoutput
    save(['Results/runs/run_' char(datetime('now','Format','MM.dd.yyyy_HH:mm:ss.SSS')) '.mat'],'results');
end
toc
