function [dFE] = fe_iterative(P,C, beta, D)
% working on the assumption that succ(i) = nodes j with p_ij > 0 (not based
% on original adjacency graphs)
% theta = beta

n = size(P,1);
dFE = rand(n);
dFE(1:n+1:end) = 0; % clear diagonal
oldError = 0;
diffError = 100;
num = 0;
error_threshold = 1e-1;

while diffError > error_threshold
    selection_set = rand(1,n) > 0;
    for i = find(selection_set)
        % find neighbors
        succ = find(P(i,:) > 0);
        dFE(i,:) = -1/beta*log(sum(P(i,succ).*exp(-beta*(C(i,succ) + dFE(succ,1:n)')),2));
        dFE(i,i) = 0;
%         for k = 1:n
%            if k == i
%                continue
% %                 dFE(i,k) = 0;
%            else
%                 dFE(i,k) = -1/beta*log(sum(P(i,succ).*exp(-beta*(C(i,succ) + dFE(succ,k)'))));
% %                 summ = 0;
% %                 for idx = 1:numel(succ)
% %                     j = succ(idx);
% %                     summ = summ + P(i,j)*exp(-beta*(C(i,j) + dFE(j,k)));
% %                 end
% %                 dFE(i,k) = (-1/beta)*log(summ);
%            end
%         end   
      %  disp(dFE(i,:))
    end
    error = norm(D - dFE,'fro');
    diffError = abs(oldError - error);
    oldError = error;
    
    num = num+1;
    fprintf('Error = %.2f, diffError = %1.3E, numIter = %d\n', error, diffError, num)
end
end

