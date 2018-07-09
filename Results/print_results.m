function print_results(results)
    fp = 1;
    datasets = fieldnames(results);
    algos = fieldnames(results.(datasets{1}));
   % max_off_dataset = max(cellfun(@numel,datasets));
    max_off_algo = max(cellfun(@numel,algos));
    M = cell(numel(datasets)+2,1);
    M{1} = sprintf(sprintf('%% %ds| ',max_off_algo),[],algos{:});
    M{2} = repmat('-',numel(M{1}),1);
    for type = 1:numel(datasets)
        str = sprintf(sprintf('%% %ds| ',max_off_algo),datasets{type});
        for alg = 1:numel(algos)
            ccr = results.(datasets{type}).(algos{alg}).ccr;
            nmi = results.(datasets{type}).(algos{alg}).nmi;
            mu_ccr = mean(ccr);
            std_ccr = std(ccr);
            mu_nmi = mean(nmi);
            std_nmi = std(nmi);
            str = [str sprintf(sprintf('%% %d.2f| ',max_off_algo),mu_nmi)];
        end
        M{type+2} = str;
    end
    fprintf(fp,'mean nmi:\n');
    fprintf(fp,'%s\n',M{:});
end