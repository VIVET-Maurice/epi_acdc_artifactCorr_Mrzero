function report = validate_experiment_current_constraint()
%VALIDATE_EXPERIMENT_CURRENT_CONSTRAINT Independent L1 and derivative checks.
nc = 32; nstates = 32; total_A = 30;
previous_rng = rng; cleanup = onCleanup(@() rng(previous_rng)); %#ok<NASGU>
rng(107);
x = .8*(2*rand(nc*nstates,1)-1);
[c,ceq,G,Geq] = experiment_current_constraint(x,nc,nstates,total_A);
assert(isempty(ceq) && isempty(Geq) && issparse(G) && ...
    isequal(size(G),[nc*nstates nstates]),'Wrong fmincon constraint interface.');
assert(all(c<0) && all(sum(abs(reshape(x,nc,nstates)),1)<total_A), ...
    'Moderate currents should be feasible.');
errors = zeros(1,4);
for test = 1:4
    direction = randn(size(x)); direction = direction/norm(direction);
    step = 1e-6;
    numerical = (experiment_current_constraint(x+step*direction,nc,nstates,total_A) ...
        -experiment_current_constraint(x-step*direction,nc,nstates,total_A))/(2*step);
    analytical = G.'*direction;
    errors(test) = norm(numerical-analytical)/max(1,norm(analytical));
end
assert(max(errors)<1e-7,'Absolute-current constraint gradient failed.');
% Cancellation of signed currents cannot bypass the absolute-current budget.
cancelled = repmat([ones(16,1);-ones(16,1)],1,nstates);
assert(all(sum(cancelled,1)==0) && ...
    all(experiment_current_constraint(cancelled(:),nc,nstates,total_A)>0), ...
    'Signed-current cancellation incorrectly bypasses the 30 A limit.');
boundary = repmat([ones(30,1);zeros(2,1)],1,nstates);
assert(all(experiment_current_constraint(boundary(:),nc,nstates,total_A)>0), ...
    'A sum equal to 30 A must not satisfy the strict limit.');
[zero_c,~,zero_G] = experiment_current_constraint(zeros(size(x)),nc,nstates,total_A);
assert(all(isfinite(zero_c)) && nnz(zero_G)==0 && all(zero_c<0), ...
    'Zero currents must have a finite, feasible constraint and gradient.');
safe = .99999*boundary;
assert(all(experiment_current_constraint(safe(:),nc,nstates,total_A)<0) && ...
    all(sum(abs(safe),1)<total_A),'A current budget below the margin should be feasible.');
report = struct('directional_gradient_relative_errors',errors, ...
    'strict_absolute_current_limit_verified',true,'zero_derivative_verified',true);
fprintf('Current constraint validation: gradient max %.3g; absolute sum and strict limit passed.\n',max(errors));
end
