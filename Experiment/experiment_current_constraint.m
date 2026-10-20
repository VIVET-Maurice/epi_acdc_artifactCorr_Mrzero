function [c,ceq,gradient_c,gradient_ceq] = experiment_current_constraint(x,nc,nstates,total_A)
%EXPERIMENT_CURRENT_CONSTRAINT Conservative differentiable per-line L1 bound.
%   sum(sqrt(I.^2+epsilon^2)) >= sum(abs(I)); hence feasibility implies
%   the requested absolute-current budget. A 0.1 mA numerical margin keeps
%   the actual sum strictly below total_A. No auxiliary current variables.
epsilon_A = 1e-6;
margin_A = 1e-4;
validateattributes(total_A,{'numeric'},{'real','scalar','finite','>',margin_A});
currents = reshape(x,nc,nstates);
smooth_abs = sqrt(currents.^2+epsilon_A^2);
c = sum(smooth_abs,1).'- (total_A-margin_A);
ceq = [];
if nargout>2
    % fmincon expects variables-by-constraints, one sparse block per line.
    rows = (1:nc*nstates).';
    columns = repelem((1:nstates).',nc);
    gradient_c = sparse(rows,columns,currents(:)./smooth_abs(:),nc*nstates,nstates);
    gradient_ceq = [];
end
end
