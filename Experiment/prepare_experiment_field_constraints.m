function [A, b, info] = prepare_experiment_field_constraints(ctx, field_limit_Hz, total_current_A)
%PREPARE_EXPERIMENT_FIELD_CONSTRAINTS Remove provably redundant field rows.
%   Assumes EACH state's currents obey |I_j| <= Imax_A(j) and
%   sum(abs(I)) <= TOTAL_CURRENT_A. Both bounds must remain in the solver.
%   The exact maximum |Brow*I| under those bounds is the fractional-knapsack
%   allocation of the L1 budget to descending abs(Brow) coefficients.
%   A row is removed only when this maximum plus conservative roundoff
%   margin is strictly smaller than FIELD_LIMIT_HZ. Borderline rows remain.
%   No voxel downsampling, approximate field basis or changed physics.

validateattributes(field_limit_Hz,{'numeric'}, {'real','finite','scalar','positive'});
validateattributes(total_current_A,{'numeric'}, {'real','finite','scalar','nonnegative'});
validateattributes(ctx.B_quality,{'numeric'}, {'real','finite','2d'});
validateattributes(ctx.nstates,{'numeric'}, {'real','finite','scalar','integer','positive'});
B = double(ctx.B_quality);
[nvoxels,nchannels] = size(B);
assert(nchannels==ctx.nc, 'Field basis/channel count disagree.');
limits = double(ctx.Imax_A(:));
if isscalar(limits), limits = repmat(limits,nchannels,1); end
validateattributes(limits,{'numeric'}, {'real','finite','positive','size',[nchannels 1]});

[coefficients,order] = sort(abs(B),2,'descend');
capacities = reshape(limits(order(:)),size(order));
spent_before = [zeros(nvoxels,1),cumsum(capacities(:,1:end-1),2)];
remaining = max(double(total_current_A)-spent_before,0);
allocation = min(capacities,remaining);
maximum_field = sum(coefficients.*allocation,2);

% The margin also covers cumsum/cancellation in the budget allocation,
% not merely the final dot product. Using the full box bound as a scale
% makes removal conservative even when the L1 bound is substantially lower.
box_bound = abs(B)*limits;
scale = max([ones(nvoxels,1),box_bound,maximum_field, ...
    repmat(double(field_limit_Hz),nvoxels,1)],[],2);
margin = 64*eps('double')*max(1,nchannels).*scale;
keep = ~(maximum_field+margin<double(field_limit_Hz));
A = kron(speye(ctx.nstates),sparse(B(keep,:)));
A = [A;-A];
b = double(field_limit_Hz)*ones(size(A,1),1);
if isempty(maximum_field), maximum = 0; else, maximum = max(maximum_field); end
info = struct('nQualityVoxels',nvoxels,'nKeptVoxels',nnz(keep), ...
    'rowsBefore',2*nvoxels*ctx.nstates,'rowsAfter',size(A,1), ...
    'maxboxL1Field_Hz',maximum,'totalCurrent_A',double(total_current_A), ...
    'method','Exact box-plus-L1 maximum; conservative strict redundancy test');
end
