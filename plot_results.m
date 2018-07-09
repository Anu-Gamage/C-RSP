function plot_results(c_array,params)
    algos = fieldnames(c_array);
    cs = cellfun(@(x) str2double(x(2:end)), fieldnames(c_array.(algos{1})));
    figure();
    hold on;
    axis([-inf inf 100/params(2) 100]);
    fig1 = gca;
    figure();
    hold on;
    fig2 = gca;
    for ii = 1:numel(algos)
        mat = cell2mat(struct2cell(c_array.(algos{ii})));
        ccrs = mat(1:2:end,:);
        nmis = mat(2:2:end,:);
        mu_ccr = mean(ccrs,2);
        mu_nmi = mean(nmis,2);
        std_ccr = std(ccrs,[],2);
        std_nmi = std(nmis,[],2);
        errorbar(fig1,cs,mu_ccr,std_ccr);
        errorbar(fig2,cs,mu_nmi,std_nmi);
    end
    legend(fig1,algos);
    xlabel(fig1,'average degree (c)');
    ylabel(fig1,'CCR');
    saveas(fig1,sprintf('Results/sbm/sbm_sparsity_ccr_N%dk%dm%d.png',params(1),params(2),params(3)));
    legend(fig2,algos);
    xlabel(fig2,'average degree (c)');
    ylabel(fig2,'NMI');
    saveas(fig2,sprintf('Results/sbm/sbm_sparsity_nmi_N%dk%dm%d.png',params(1),params(2),params(3)));
end