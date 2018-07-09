% Main test file for MultiNMF method
% Anuththari Gamage
% 3/30/2018
clear;clc;close all

n = 100;                        % no. of nodes 
k = 2;                          % no. of clusters
m_array = [1,2];              % no. of layers
c = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0,15.0, 20.0];               % Varying node degree
lambda = 0.9;

do_plot = 0;                     % To plot data matrices
do_result_plot = 1;              % To plot results
num_runs = 2;
ccr_array = zeros(num_runs, numel(c), numel(m_array));
nmi_array = zeros(num_runs, numel(c), numel(m_array));

options = [];
options.maxIter = 75;
options.error = 1e-6;
options.nRepeat = 10;
options.minIter = 30;
options.meanFitRatio = 0.1;
options.rounds = 30;
options.WeightMode='Binary';
options.varWeight = 0;
options.kmeans = 1;
options.Gaplpha=10;        % Change to 100?                      
options.alpha=10;
options.delta = 0.1;
options.beta = 0;
options.gamma = 2;
options.K = k;


for runs = 1:num_runs
    % Generate adjacency tensor of test data
    data = cell(1,numel(c));   
    for i = 1:numel(c)
        [data{i}, labels] = mlsbm_gen(n,k,max(m_array), c(i), lambda);
    end
    for layers = 1:numel(m_array)        % Varying no. of layers
        m = m_array(layers);
        acc_val = zeros(1, numel(c));
        nmi_val = zeros(1, numel(c));
        for degree = 1:numel(c)         % Varying node degree
            A = cell(1,layers);
            for i = 1:m                 % Select relevant tensor
                A{i} = data{degree}{i};
            end
            fprintf('Variable %d processing:\n', degree)
            if do_plot
               figure; 
               for i = 1:m
                  subplot(1,m,i);spy(A{i}); title(sprintf('Layer %d', i))
               end
            end
           
            options.alphas = ones(1,m);
            [acc_val(degree), nmi_val(degree)] = GMultiNMF(A, k, A, labels, options);
            fprintf('CCR: %.2f\n', acc_val(degree))
            fprintf('NMI: %.2f\n\n', nmi_val(degree))
        end
        ccr_array(runs,:, layers) = acc_val;
        nmi_array(runs,:, layers) = nmi_val;
    end
end
% save('crsp_ccr.mat', 'ccr_array')
% save('crsp_nmi.mat', 'nmi_array')

if do_result_plot
 for i = 1:numel(m_array)
        avg_ccr = mean(ccr_array(:,:,i));
        avg_nmi = mean(nmi_array(:,:,i));
        std_ccr = std(ccr_array(:,:,i));
        std_nmi = std(nmi_array(:,:,i));
        figure;yyaxis left; errorbar(c, avg_ccr, std_ccr); ylabel('CCR'); ylim([50,100])
        hold on;yyaxis right; errorbar(c, avg_nmi, std_nmi); ylabel('NMI'); ylim([0,1])
        title(sprintf('MultiNMF: Nodes = %d, Clusters = %d, Layers = %d', n, k,m_array(i)))
        xlabel('c'); 
        legend('CCR', 'NMI','Location','SouthEast')
 end
end

