% Preprocesses 3sources data and constructs affinity matrix 
% Anuththari Gamage
% 6/29/2018

    
sources = cell(1,3);        % term frequencies in each document
ids = cell(1,3);            % document ids

sources{1} = load('3sources_bbc.mtx');              
sources{2} = load('3sources_guardian.mtx');
sources{3} = load('3sources_reuters.mtx');
ids{1} = load('3sources_bbc.docs');
ids{2} = load('3sources_guardian.docs');
ids{3} = load('3sources_reuters.docs');

% Format of .mtx : #term id    #document id    #frequency of term in document
for i = 1:3
    sources{i} = sources{i}(2:end,:); % trim metadata in first row
    sources{i}(:,2) = ids{i}(sources{i}(:,2)); % Replace doc id with global doc ids
end

% identifies the global ids of the 169 overlapping documents in the three sources
overlap = intersect(ids{1}, intersect(ids{2}, ids{3}));  
n = 169;

% Remove non-overlapping articles and sort by doc id
for i = 1:3
    sources{i} = sortrows(sources{i}(ismember(sources{i}(:,2),overlap),:),2);
end

sources_data = cell(3,1); % adjacency tensor, cost tensor, labels
sources_data{1} = cell(3,1); 
sources_data{2} = cell(3,1);

% Construct affinity matrix/cost matrix for each layer
for layer = 1:3
    num_terms = max(sources{layer}(:,1)); % total words in all documents in source 
    features = zeros(n, num_terms);  % feature matrix of term frequencies 
    for ii = 1:n
        inds = sources{layer}(sources{layer}(:,2) == ii,1);
        features(ii,inds) = sources{layer}(inds,3);
    end
    
    % Create Gaussian Kernel 
    S = squareform(pdist(features)); % Euclidean distances
    std = median(S(:));  % Standard deviation of kernel
    G = exp(-(S.^2)/(2*std^2)); % Gaussian kernel/Similarity graph
    
    sources_data{1}{layer} = G;
    
    % Create cost matrix 
    sources_data{2}{layer} = 1./G;
end
    
load('sources_labels.mat')          % list of cluster labels generated using '3sources.disjoint.clist'
sources_data{3} = sources_labels;

save('sources_data.mat', 'sources_data')


