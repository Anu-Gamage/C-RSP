function uci = readuci()
fields = {'fac','fou','mor','pix','zer'};

uci = struct('A',[],'labels',[]);
for name = 1:numel(fields)
    attr = load(sprintf('Datasets/uci/mfeat-%s',fields{name}));
    A = pdist2(attr,attr);
    sigma = median(A(:));
    uci.A.(fields{name}) = exp(-A.^2/(2*sigma^2));
end
vec = @(x) x(:);
uci.labels = vec(repmat(1:10,200,1));
end