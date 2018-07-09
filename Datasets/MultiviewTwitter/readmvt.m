function mvt = readmvt()
% Convert data matrices to sparse .mat
% Anuththari Gamage 3/25/2018
% modified Brian Rappaport, 6/30/2018

    datasets = {'politicsuk','politicsie','rugby','football','olympics'};
    datatypes = {'follows','mentions','retweets'};
    mvt = struct;
    for jj = 1:numel(datasets)
        dataset = datasets{jj};
        
        % replace ids with node index        
        fp = fopen(sprintf('Datasets/MultiviewTwitter/%s/%s.communities',dataset,dataset));
        comms = textscan(fp,'%s');
        comms = comms{1}(2:2:end);
        comms = cellfun(@str2num,comms,'UniformOutput',false);
        changes = cumsum(cellfun(@numel,comms));
        labels = zeros(1,changes(end));
        changes = [1; changes(1:end-1)+1];
        labels(changes) = 1;
        labels = cumsum(labels)';
        comms = cell2mat(comms');
        fclose(fp);

        conn = 1:numel(labels);
        set = struct();
        for ii = 1:numel(datatypes)
            data = load((sprintf('Datasets/MultiviewTwitter/%s/%s-%s.mtx', dataset, dataset, datatypes{ii})));
            data = data(2:end,:); % remove header
            rows = data(:,1);
            cols = data(:,2);
            elems = [rows cols];

            [toreplace, bywhat] = ismember(elems, comms);
            elems(toreplace) = bywhat(toreplace);

            A = sparse(elems(1:end/2),elems(end/2+1:end),data(:,3),numel(labels),numel(labels));
            conn = intersect(find(any(A,2)),conn);
            set.A.(datatypes{ii}) = A;
        end
        for ii = 1:numel(datatypes)
            A = set.A.(datatypes{ii});
            A = full(A(conn,conn));
            A(A <= 1e-12) = 1e-12;
            set.A.(datatypes{ii}) = A;
        end
        set.labels = labels(conn);
        mvt.(datasets{jj}) = set;
    end
end