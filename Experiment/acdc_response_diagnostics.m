function diagnostics = acdc_response_diagnostics(currents_A, ctx, field_limit_Hz, total_current_limit_A)
%ACDC_RESPONSE_DIAGNOSTICS Compare DAC commands and predicted real currents.
%   All times are seconds after the RF center, matching the forward model.
%   ADC arrays follow ctx.time_s(:), not chronological sorting. Field/current
%   limits are checked at ADC samples only, not continuously between samples.
%   The two images use IDENTICAL DAC commands; only the response model differs.
%   CSV exports must continue to use currents_A, never these filtered currents.
if nargin<3, field_limit_Hz = Inf; end
if nargin<4, total_current_limit_A = Inf; end
validateattributes(currents_A,{'numeric'}, ...
    {'real','finite','size',[ctx.nc ctx.nstates]});
currents_A = double(currents_A);
time = double(ctx.time_s(:));
starts = double(ctx.line_start_s(:));
ends = double(ctx.line_end_s(:));
if isfield(ctx,'current_response')
    response = ctx.current_response;
else
    response = struct('enabled',false,'tau_s',repmat(133e-6/log(100),ctx.nc,1));
end
command_weights = double(time>=starts.' & time<ends.');
commanded_currents = currents_A*command_weights.';
commanded_integrals = currents_A*double(ctx.H_s).';
if response.enabled
    [actual_currents,integrated_currents] = apply_response( ...
        currents_A,ctx.R_response,ctx.H_response_s);
else
    actual_currents = commanded_currents;
    integrated_currents = commanded_integrals;
end

% Each Cartesian line identifies its chronological DAC state explicitly.
line_center_time = zeros(ctx.nstates,1);
for state = 1:ctx.nstates
    selected = double(ctx.line_index(:))==state;
    assert(any(selected),'Every current state needs imaging ADC samples.');
    local_times = time(selected);
    [~,center_index] = min(abs(local_times-median(local_times)));
    line_center_time(state) = local_times(center_index);
end
line_currents = sample_currents(currents_A,starts,ends,line_center_time,response);

% Zoom around the largest commanded channel jump, preserving ALL earlier
% transitions when evaluating the real currents in this local time window.
jumps = [currents_A(:,1),diff(currents_A,1,2)];
[~,jump_index] = max(abs(jumps(:)));
[zoom_channel,zoom_state] = ind2sub(size(jumps),jump_index);
zoom_radius = 5*max(double(response.tau_s));
zoom_time = linspace(max(0,starts(zoom_state)-zoom_radius), ...
    starts(zoom_state)+zoom_radius,512).';
zoom_command = currents_A*double(zoom_time>=starts.' & zoom_time<ends.').';
zoom_actual = sample_currents(currents_A,starts,ends,zoom_time,response);
% Resolve every transition over the complete acquisition, including the
% final decay when the last command window has a finite end.
[waveform_actual,waveform_command,waveform_time] = ...
    acdc_sample_current_waveform(currents_A,starts,ends,[],response);
actual_quality_field = double(ctx.B_quality)*actual_currents;
actual_current_sum = sum(abs(actual_currents),1);
maximum_actual_field = max(abs(actual_quality_field),[],'all');
actual_channel_limits_met = all(all(abs(actual_currents)<=ctx.Imax_A+1e-8));

% A prepared accelerated runtime owns base signals in contiguous blocks.
% Recover their original ADC order once here for a CPU diagnostic twin.
plain_ctx = ctx;
plain_ctx.B = cpu_array(ctx.B);
plain_ctx.reference = cpu_array(ctx.reference);
if isempty(plain_ctx.base_signal)
    plain_ctx.base_signal = complex(zeros(size(ctx.B,1),numel(time)));
    for block = 1:numel(ctx.blocks)
        b = ctx.blocks(block);
        plain_ctx.base_signal(:,b.OriginalSamples) = cpu_array(b.BaseSignal);
    end
end
instantaneous_ctx = plain_ctx;
instantaneous_ctx.current_response.enabled = false;
[instantaneous_image,instantaneous_kdata] = forward_model_experiment_acdc(currents_A,instantaneous_ctx);
if response.enabled
    [response_image,response_kdata] = forward_model_experiment_acdc(currents_A,plain_ctx);
else
    response_image = instantaneous_image;
    response_kdata = instantaneous_kdata;
end
instantaneous_mse = sum((abs(instantaneous_image(:))-abs(plain_ctx.reference(:))).^2) ...
    /ctx.reference_energy;
response_mse = sum((abs(response_image(:))-abs(plain_ctx.reference(:))).^2) ...
    /ctx.reference_energy;
step_time = linspace(0,1.3*max(response.tau_s)*log(100),256).';
step_response = -expm1(-step_time.'./double(response.tau_s(:)));
diagnostics = struct('adc_time_s',time, ...
    'commanded_currents_at_adc_A',commanded_currents, ...
    'actual_currents_at_adc_A',actual_currents, ...
    'commanded_integrated_currents_to_adc_As',commanded_integrals, ...
    'integrated_currents_to_adc_As',integrated_currents, ...
    'line_center_time_s',line_center_time, ...
    'generated_field_at_line_center_Hz', ...
        reshape(double(ctx.B_full)*line_currents,[ctx.object_size ctx.nstates]), ...
    'dense_time_s',zoom_time,'dense_commanded_currents_A',zoom_command, ...
    'dense_actual_currents_A',zoom_actual, ...
    'dense_transition_index',zoom_state,'dense_transition_channel',zoom_channel, ...
    'waveform_time_s',waveform_time, ...
    'waveform_commanded_currents_A',waveform_command, ...
    'waveform_actual_currents_A',waveform_actual, ...
    'step_time_s',step_time,'unit_step_response_A',step_response, ...
    'step_reach99_time_s',double(response.tau_s(:))*log(100), ...
    'maximum_actual_quality_field_Hz',maximum_actual_field, ...
    'actual_adc_field_limits_met',maximum_actual_field<=field_limit_Hz+1e-8, ...
    'absolute_actual_current_sum_at_adc_A',actual_current_sum, ...
    'maximum_actual_current_sum_at_adc_A',max(actual_current_sum), ...
    'actual_adc_channel_limits_met',actual_channel_limits_met, ...
    'actual_adc_current_sum_limits_met',all(actual_current_sum<total_current_limit_A), ...
    'instantaneous_image',instantaneous_image,'instantaneous_kdata',instantaneous_kdata, ...
    'instantaneous_relative_mse',instantaneous_mse, ...
    'response_image',response_image,'response_kdata',response_kdata, ...
    'response_relative_mse',response_mse, ...
    'limit_check_domain','ADC samples only; commanded bounds remain optimization constraints');
end

function currents = sample_currents(commands,starts,ends,times,response)
currents = acdc_sample_current_waveform(commands,starts,ends,times,response);
end

function [currents,integrals] = apply_response(commands,response_weights,integral_weights)
currents = zeros(size(commands,1),size(response_weights,1));
if nargout>1, integrals = zeros(size(currents)); end
for channel = 1:size(commands,1)
    currents(channel,:) = commands(channel,:)*double(response_weights(:,:,channel)).';
    if nargout>1
        integrals(channel,:) = commands(channel,:)*double(integral_weights(:,:,channel)).';
    end
end
end

function value = cpu_array(value)
if isa(value,'gpuArray'), value = gather(value); end
end
