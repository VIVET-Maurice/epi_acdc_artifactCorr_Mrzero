function report = validate_experiment_field_constraints()
%VALIDATE_EXPERIMENT_FIELD_CONSTRAINTS Independent small-polytope checks.
%   Enumerates feasible vertices in every orthant for a three-channel
%   synthetic fixture. No calibration/current/output file is created.
previous_rng = rng;
cleanup = onCleanup(@() rng(previous_rng)); %#ok<NASGU>
rng(748);
ctx = struct('B_quality',[7 4 1;-2 9 -3;0 0 0;4 4 4;1e-14 0 0], ...
    'nc',3,'nstates',4,'Imax_A',[1;2;3]);
budgets = [0 .5 2.5 6 8];
field_limits = [.25 8 15 24 100];
max_bound_error = 0;
comparisons = 0;
removed_any = false;
for budget = budgets
    vertices = feasible_vertices(ctx.Imax_A,budget);
    measured_bounds = max(abs(ctx.B_quality*vertices),[],2);
    [~,~,info] = prepare_experiment_field_constraints(ctx,100,budget);
    max_bound_error = max(max_bound_error,abs(info.maxboxL1Field_Hz-max(measured_bounds)));
    for limit = field_limits
        [A,b,info] = prepare_experiment_field_constraints(ctx,limit,budget);
        assert(issparse(A) && size(A,2)==ctx.nc*ctx.nstates && ...
            numel(b)==size(A,1) && info.rowsAfter==2*info.nKeptVoxels*ctx.nstates, ...
            'Unexpected constraint dimensions.');
        guaranteed_removed = measured_bounds+1e-11<limit;
        borderline = abs(measured_bounds-limit)<1e-11;
        assert(info.nKeptVoxels>=nnz(borderline), 'A borderline maximum was removed.');
        assert(info.nKeptVoxels==nnz(~guaranteed_removed) || ...
            all(measured_bounds<limit), 'Incorrect row-redundancy count.');
        removed_any = removed_any || info.rowsAfter<info.rowsBefore;

        % Test feasible vertices and independently sampled box/L1 currents.
        random_currents = ctx.Imax_A.*(2*rand(3,400)-1);
        sums = sum(abs(random_currents),1);
        random_currents = random_currents.*min(1,budget./max(sums,realmin));
        candidates = [vertices random_currents];
        for column = 1:size(candidates,2)
            currents = repmat(candidates(:,column),1,ctx.nstates);
            % Different signed/permuted amplitudes across states remain
            % feasible and exercise kron's channels-then-states layout.
            currents(:,2:2:end) = -.7*currents(:,2:2:end);
            full_feasible = all(abs(ctx.B_quality*currents)<=limit+1e-10,'all');
            reduced_feasible = all(A*currents(:)<=b+1e-10);
            assert(full_feasible==reduced_feasible, 'Removal changed feasible current states.');
            comparisons = comparisons+1;
        end
    end
end
assert(max_bound_error<1e-11 && removed_any, 'Exact bounds/removal were not verified.');

% Known analytic vertices give individual bounds, not only a global max.
one = ctx; one.nstates = 1;
for row = 1:size(ctx.B_quality,1)
    one.B_quality = ctx.B_quality(row,:);
    vertices = feasible_vertices(ctx.Imax_A,2.5);
    exact = max(abs(one.B_quality*vertices));
    [~,~,info] = prepare_experiment_field_constraints(one,100,2.5);
    assert(abs(info.maxboxL1Field_Hz-exact)<1e-12,'Per-row knapsack maximum is wrong.');
end
one.B_quality = [7 4 1];
[~,~,info] = prepare_experiment_field_constraints(one,100,2.5);
assert(abs(info.maxboxL1Field_Hz-13)<1e-13,'Known maximum 7*1+4*1.5 is incorrect.');
[A,b,info] = prepare_experiment_field_constraints(ctx,100,2.5);
assert(isequal(size(A),[0 ctx.nc*ctx.nstates]) && isempty(b) && info.nKeptVoxels==0, ...
    'An entirely redundant field system must become zero rows.');
one.B_quality = [7 4 1];
[A,~,info] = prepare_experiment_field_constraints(one,13,2.5);
assert(info.nKeptVoxels==1 && size(A,1)==2,'An exact boundary must be retained.');

report = struct('status','passed','maximum_bound_error_Hz',max_bound_error, ...
    'feasible_current_comparisons',comparisons,'all_removed_dimensions_verified',true, ...
    'borderline_retention_verified',true,'synthetic_fixture',true);
fprintf('Exact field redundancy: bound error %.3g Hz; %d feasible-current comparisons; all-removed/borderline passed.\n', ...
    max_bound_error,comparisons);
end

function vertices = feasible_vertices(capacity,budget)
% Enumerate all vertices of 0<=y<=capacity, sum(y)<=budget, then all signs.
nc = numel(capacity);
A = [eye(nc);-eye(nc);ones(1,nc)];
b = [capacity;zeros(nc,1);budget];
sets = nchoosek(1:size(A,1),nc);
positive = zeros(nc,0);
for row = 1:size(sets,1)
    equations = A(sets(row,:),:);
    if rcond(equations)<1e-12, continue; end
    candidate = equations\b(sets(row,:));
    if all(A*candidate<=b+1e-12), positive(:,end+1) = candidate; end %#ok<AGROW>
end
assert(~isempty(positive),'Polytope enumeration found no feasible vertex.');
vertices = zeros(nc,0);
for bits = 0:2^nc-1
    signs = 2*bitget(bits,1:nc).'-1;
    vertices = [vertices signs.*positive]; %#ok<AGROW>
end
end
