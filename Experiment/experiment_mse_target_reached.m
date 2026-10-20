function reached = experiment_mse_target_reached(x, value, ctx, field_limit_Hz, config)
%EXPERIMENT_MSE_TARGET_REACHED Accept a low MSE only with physical bounds met.
%   An empty TargetMSE disables this optional stopping criterion. Actual
%   channel currents and generated fields are checked only below the target;
%   the total absolute current must remain strictly below its budget.
reached = false;
if isempty(config.TargetMSE) || ~isscalar(value) || ~isfinite(value) || ...
        value>=config.TargetMSE
    return;
end
currents_A = reshape(x,ctx.nc,ctx.nstates);
reached = all(isfinite(currents_A),'all') && ...
    all(abs(currents_A)<=ctx.Imax_A+1e-8,'all') && ...
    all(sum(abs(currents_A),1)<config.TotalCurrentLimit_A) && ...
    max(abs(ctx.B_quality*currents_A),[],'all')<=field_limit_Hz+1e-8;
end
