function [currents_A, mode] = prepare_experiment_initial_currents(initial_A, ctx, total_limit_A, field_limits_Hz)
%PREPARE_EXPERIMENT_INITIAL_CURRENTS Validate the image solver's DAC seed.
%   [] gives zero commands. A channel vector repeats over every imaging
%   state; a channels-by-states matrix supplies each state's initial guess.
%   This does not change the physical current before the first command,
%   which remains zero in the temporal response model. Invalid guesses are
%   rejected before optimization without clipping, scaling or substitution.

validateattributes(initial_A,{'numeric'},{'real','finite','2d'});
validateattributes(total_limit_A,{'numeric'},{'real','finite','scalar','positive'});
validateattributes(field_limits_Hz,{'numeric'},{'real','finite','positive','vector','nonempty'});
if isempty(initial_A)
    currents_A = zeros(ctx.nc,ctx.nstates);
    mode = 'zeros';
elseif isvector(initial_A) && numel(initial_A)==ctx.nc
    currents_A = repmat(double(initial_A(:)),1,ctx.nstates);
    mode = 'channel vector repeated over imaging states';
elseif isequal(size(initial_A),[ctx.nc ctx.nstates])
    currents_A = double(initial_A);
    mode = 'explicit DAC commands';
else
    error('prepare_experiment_initial_currents:Shape', ...
        'InitialCurrents_A needs %d channel values or a %d-by-%d matrix.', ...
        ctx.nc,ctx.nc,ctx.nstates);
end
if any(any(abs(currents_A)>ctx.Imax_A))
    error('prepare_experiment_initial_currents:ChannelLimit', ...
        'Initial DAC commands exceed a channel current limit.');
end
if any(sum(abs(currents_A),1)>=total_limit_A)
    error('prepare_experiment_initial_currents:TotalLimit', ...
        'Initial DAC commands must have sum(abs(I)) strictly below TotalCurrentLimit_A in every state.');
end
if max(abs(ctx.B_quality*currents_A),[],'all')>min(field_limits_Hz)
    error('prepare_experiment_initial_currents:FieldLimit', ...
        'Initial DAC commands exceed a generated-field limit on measured-quality voxels.');
end
if all(currents_A(:)==0), mode = 'zeros'; end
end
