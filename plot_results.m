function plot_results(c_array,params)
    algos = fieldnames(c_array);
    
    cs = cellfun(@(x) str2double(x(2:end)), fieldnames(c_array.(algos{1})));
    if isfield(c_array.(algos{1}),['c' num2str(cs(1))])
        using_c = 1;
    else
        using_c = 0;
    end
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
        errorbar(fig1,cs,mu_ccr,std_ccr,'LineWidth',1.2);
        errorbar(fig2,cs,mu_nmi,std_nmi,'LineWidth',1.2);
    end
    if using_c
        xlabeltext = 'average degree (c)';
        filenamebits = 'sparsity';
        var3 = 'm';
    else
        xlabeltext = 'number of layers (m)';
        filenamebits = 'layers';
        var3 = 'c';
    end
    legend(fig1,algos);
    xlabel(fig1,xlabeltext);
    ylabel(fig1,'CCR');
    saveas(fig1,sprintf('Results/sbm/sbm_%s_ccr_N%dk%d%s%d.png',filenamebits,params(1),params(2),var3,params(3)));
    legend(fig2,algos);
    xlabel(fig2,xlabeltext);
    ylabel(fig2,'NMI');
    saveas(fig2,sprintf('Results/sbm/sbm_%s_nmi_N%dk%d%s%d.png',filenamebits,params(1),params(2),var3,params(3)));
end