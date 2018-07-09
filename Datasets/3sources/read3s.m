function three = read3s()
    three = struct();

    datatypes = {'bbc','guardian','reuters'};
    m = numel(datatypes);
    ids = cell(1,m);
    overall_id = 1:1000; %TODO magic number but don't see a pretty way to take it out
    for ii = 1:m
        ids{ii} = load(sprintf('Datasets/3sources/3sources_%s.docs',datatypes{ii}));
        overall_id = intersect(overall_id, ids{ii});
    end
    n = numel(overall_id);
    fp = fopen('Datasets/3sources/3sources.disjoint.clist');
    comms = textscan(fp,'%s');
    comms = comms{1}(2:2:end);
    comms = cellfun(@str2num,comms,'UniformOutput',false);
    comms = cellfun(@(x) intersect(x,overall_id)',comms,'un',0);
    changes = cumsum(cellfun(@numel,comms));
    labels = zeros(1,changes(end));
    changes = [1; changes(1:end-1)+1];
    labels(changes) = 1;
    labels = cumsum(labels)';
    comms = cell2mat(comms')';
    toGet = sortrows([comms labels]);
    toGet = sortrows([toGet (1:n)'],2);
    reorder = toGet(:,3);
    fclose(fp);
    three.labels = labels;
    for ii = 1:m
        data = load(sprintf('Datasets/3sources/3sources_%s.mtx', datatypes{ii}));
        f = data(1,1); % total words in all documents in source
        data = data(2:end,:); % remove header
        data(:,2) = ids{ii}(data(:,2));
        data = sortrows(data(ismember(data(:,2),overall_id),:),2);

        features = zeros(n,f);  % feature matrix of term frequencies 
        for jj = 1:n
            inds = data(data(:,2) == (jj),1);
            features((jj),inds) = data(inds,3);
        end

        S = squareform(pdist(features)); % Euclidean distances
        std = median(S(:));  % Standard deviation of kernel
        G = exp(-(S.^2)/(2*std^2)); % Gaussian kernel/Similarity graph
        three.A.(datatypes{ii}) = G;
    end
end