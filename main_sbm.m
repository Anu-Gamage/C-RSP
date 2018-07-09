% Test file for sbm, varying sparsity

close all
tic

N = 500;
lambda = 0.9;

% Algorithms used for comparison: 1 = C-RSP, 2 = SC-ML, 3 = CoregSC
alg_names = {'CRSP', 'SCML', 'CSC'};
algs = 1:3;

results = struct;
num_reps = 10;

k_array = 3;
m_array = 3;
c_array = 2:2:20;
for ii = 1:numel(k_array)
    k = k_array(ii);
    for jj = 1:numel(m_array)
        m = m_array(jj);
        for kk = 1:numel(c_array)
            cin = c_array(kk);
            [A,labels] = mlsbm_gen(N,k,m,cin,lambda);
            n = size(A{1},1);
            if size(labels,1) ~= 1
                labels = labels';
            end
            for ll = algs
                acc_val = zeros(1,num_reps);
                nmi_val = zeros(1,num_reps);
                for mm = 1:num_reps
                    switch alg_names{ll}
                        case 'CRSP'
                            b = 0.02;                       % inverse temperature parameter
                            [acc_val(mm),nmi_val(mm),~] = CRSP(A,n,k,m,b,labels);
                        case 'SCML'
                            lambda_scml = 0.5;              % regularization parameter for SC-ML
                            [acc_val(mm),nmi_val(mm),~] = SCML(A,k,lambda_scml,labels);
                        case 'CSC'
                            lambda_coreg = 0.05;            % Co-regularization parameter for PCSC/CCSC
                            num_iter = 1;                   % no. of iterations for PCSC
                            sigma = zeros(1,m);
%                             [acc_val,nmi_val,final_labels] = spectral_pairwise_multiview(A,m,k,sigma,lambda_coreg,labels,num_iter);
                            [acc_val(mm),nmi_val(mm),~] = spectral_centroid_multiview(A,m,k,sigma,repmat(lambda_coreg,1,m),labels',num_iter);
                        otherwise
                    end
                end
                results.(['N' num2str(N) 'k' num2str(k) 'm' num2str(m)]).(alg_names{ll}).(['c' num2str(cin)]) = [acc_val;nmi_val];
            end
        end
        plot_results(results.(['N' num2str(N) 'k' num2str(k) 'm' num2str(m)]),[N k m]);
    end
end
outputfile = ['Results/sbm/sbm_' char(datetime('now','Format','MM.dd.yyyy_HH:mm:ss.SSS')) '.mat'];
save(outputfile,'results');
toc
