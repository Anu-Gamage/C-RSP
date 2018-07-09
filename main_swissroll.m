% Test file to display swiss roll
close all

sw = swiss_roll(10);
A = struct2cell(sw.A);
labels = 3;
k = labels;
m = numel(A);
n = size(A{1},1);

figure(1);
scatter3(sw.gt(:,1),sw.gt(:,2),sw.gt(:,3),[],1:n);
view([-165, 7]);

alg_names = {'CRSP','SCML','min','max'};
algs = 1:4;

for ii = algs
    switch alg_names{ii}
        case 'CRSP'
            b = 0.02;                    % inverse temperature parameter
            dRSP = CRSP(A,n,k,m,b);
            emb = cmdscale(dRSP,3);
        case 'SCML'
            lambda_scml = 0.5;              % regularization parameter for SC-ML
            emb = SCML(A,k,lambda_scml);
        case 'min'
            AA = reshape(cell2mat(A)',n,n,m);
            dist = min(AA,[],3);
            emb = cmdscale(dist,3);
        case 'max'
            AA = reshape(cell2mat(A)',n,n,m);
            dist = max(AA,[],3);
            emb = cmdscale(dist,3);
        otherwise
    end
    figure(1+ii);
    scatter3(emb(:,1),emb(:,2),emb(:,3),[],1:n);
    view([-165, 7]);
end
