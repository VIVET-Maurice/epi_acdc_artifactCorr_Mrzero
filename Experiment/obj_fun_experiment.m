function [value, gradient] = obj_fun_experiment(x, ctx)
%OBJ_FUN_EXPERIMENT Relative magnitude-image MSE and its analytic gradient.
%   The reference is the same transverse object with ideal zero B0. This is
%   a correction target, not a measured baseline B0 map. Relative MSE makes
%   a common arbitrary receiver gain cancel from the objective.
currents_A = reshape(x,ctx.nc,ctx.nstates);
image = forward_model_experiment_acdc(currents_A,ctx);
amplitude = abs(image);
residual = amplitude-abs(ctx.reference);
value = sum(residual(:).^2)/ctx.reference_energy;
if nargout>1
    lambda_image = complex(zeros(ctx.image_size));
    nonzero = amplitude>0;
    lambda_image(nonzero) = 2*residual(nonzero).*image(nonzero) ...
        ./amplitude(nonzero)/ctx.reference_energy;
    % At exactly zero magnitude choose zero as a generalized derivative.
    [~,~,gradient_I] = forward_model_experiment_acdc(currents_A,ctx,lambda_image);
    gradient = gradient_I(:);
end
end
