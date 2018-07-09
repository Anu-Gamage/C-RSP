function [final_labels,ccr_val,nmi_val] = postproc(emb,k,ground_truth)
    % Clustering using k-Means and Linear Sum Assignment
    est_labels = kmeans(real(emb), k)'; % 1 x n
    n = numel(ground_truth);
    if size(ground_truth,1) ~= 1 % is now redundant but better safe than sorry
        ground_truth = ground_truth'; % 1 x n
    end
    
    C = confusionmat(ground_truth, est_labels);
    new_labels = munkres(-C);
    final_labels = zeros(1,n);
    for i = 1:length(new_labels)
        final_labels(est_labels == new_labels(i)) = i;
    end
    % Compute accuracy
    ccr_val = 100*sum(ground_truth == final_labels)/n;
    nmi_val = nmi(ground_truth, final_labels);
end