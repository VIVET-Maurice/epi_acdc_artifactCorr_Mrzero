function report = validate_acdc_current_response()
%VALIDATE_ACDC_CURRENT_RESPONSE Independent tests of the ACDC response model.
%   Uses in-memory synthetic data only. Does not save currents or control
%   hardware. Checks the response, phase integral and optimization adjoint.
previous_rng = rng;
rng_cleanup = onCleanup(@() rng(previous_rng)); %#ok<NASGU>
rng(133);
tau = 133e-6/log(100);

% The experimental specification fixes tau through one independent point.
times = [0; tau; 133e-6; 8*tau];
[H,R] = acdc_current_response(0,Inf,times,tau);
expected = 1-exp(-times/tau);
report.step_relative_error = norm(R(:)-expected)/norm(expected);
report.step_at_133_us = R(3);
assert(abs(R(3)-.99)<1e-14 && report.step_relative_error<1e-14, ...
    'The unit step does not reach 99 percent at 133 us.');
expected_integral = times-tau*(1-exp(-times/tau));
assert(max(abs(H(:)-expected_integral))<1e-18, 'Incorrect unit-step phase integral.');

% Closely spaced commands, negative steps, gaps and a final zero command.
% An independent recurrence uses the attained current at EVERY transition.
starts = tau*[0; .5; 1; 2.1];
ends = tau*[.5; 1; 1.5; 2.6];
commands = [1; -.4; .8; -.2];
times = tau*[0; .2; .5; .7; 1; 1.3; 1.5; 1.8; 2.1; 2.4; 2.6; 3.2];
[H,R] = acdc_current_response(starts,ends,times,tau);
actual = R*commands;
independent = actual_from_recurrence(times,starts,ends,commands,tau);
report.history_absolute_error_A = max(abs(actual-independent));
assert(report.history_absolute_error_A<1e-14, ...
    'The response lost the actual current inherited from earlier transitions.');
assert(abs(actual(3)-(1-exp(-.5)))<1e-14 && actual(end)~=0, ...
    'Current must remain continuous at changes and decay after the final window.');

% Numerical integration of the independent recurrence checks the phase.
% Include command boundaries and acquisition times in the quadrature grid.
dense_times = unique([linspace(0,times(end),60001).'; starts; ends; times]);
dense_current = actual_from_recurrence(dense_times,starts,ends,commands,tau);
numerical_integral = cumtrapz(dense_times,dense_current);
[~,sample_index] = ismember(times,dense_times);
report.integral_absolute_error_As = max(abs(H*commands-numerical_integral(sample_index)));
assert(report.integral_absolute_error_As<2e-12, ...
    'The analytic accumulated current differs from independent quadrature.');

% A nonzero RF center removes only pre-RF phase, not pre-RF current history.
rf_time = tau*.6;
adc_times = times(times>=rf_time);
[H_rf,R_rf] = acdc_current_response(starts,ends,adc_times,tau,rf_time);
[H_at_rf,~] = acdc_current_response(starts,ends,rf_time,tau);
[H_from_zero,R_from_zero] = acdc_current_response(starts,ends,adc_times,tau);
assert(max(abs(H_rf*commands-(H_from_zero-H_at_rf)*commands))<1e-17 && ...
    isequal(R_rf,R_from_zero), 'An RF center must reset phase, not real current.');

% Channels can have different time constants; timing can arrive unsorted.
tau_channels = tau*[.5; 1; 2];
channel_times = flipud([0; tau; 133e-6]);
[~,R_channels] = acdc_current_response(0,Inf,channel_times,tau_channels);
report.per_channel_step_absolute_error = 0;
for channel = 1:numel(tau_channels)
    expected = 1-exp(-channel_times/tau_channels(channel));
    report.per_channel_step_absolute_error = max(report.per_channel_step_absolute_error, ...
        max(abs(R_channels(:,1,channel)-expected)));
end
assert(report.per_channel_step_absolute_error<1e-14, 'Per-channel tau is not applied.');
[H_before,R_before] = acdc_current_response(2*tau,4*tau,[0; tau; 2*tau],tau);
assert(all(H_before==0) && all(R_before==0), 'The response must be causal.');
[H_tiny,~] = acdc_current_response(starts,ends,times,1e-12);
instantaneous_integral = max(0,min(times,ends.')-starts.');
assert(max(abs(H_tiny(:)-instantaneous_integral(:)))<=1.01e-12, ...
    'The response does not approach instantaneous transitions as tau tends to zero.');

% Reject nonphysical metadata instead of silently changing the acquisition.
expect_failure(@() acdc_current_response(0,Inf,1e-4,0));
expect_failure(@() acdc_current_response(0,Inf,1e-4,-tau));
expect_failure(@() acdc_current_response(0,Inf,1e-4,NaN));
expect_failure(@() acdc_current_response([0; tau],[2*tau; 3*tau],3*tau,tau));
expect_failure(@() acdc_current_response([tau; 0],[2*tau; 3*tau],3*tau,tau));
expect_failure(@() acdc_current_response(0,Inf,1e-4,tau,2e-4));

[experiment,calibration] = synthetic_epi();
legacy = prepare_experiment_optimizer(experiment,calibration,1);
disabled_experiment = experiment;
disabled_experiment.current_response = struct('enabled',false,'tau_s',tau);
disabled = prepare_experiment_optimizer(disabled_experiment,calibration,1);
x = .02*randn(legacy.nc*legacy.nstates,1);
currents = reshape(x,legacy.nc,legacy.nstates);
[legacy_image,legacy_k] = forward_model_experiment_acdc(currents,legacy);
[disabled_image,disabled_k] = forward_model_experiment_acdc(currents,disabled);
[legacy_value,legacy_gradient] = obj_fun_experiment(x,legacy);
[disabled_value,disabled_gradient] = obj_fun_experiment(x,disabled);
assert(isequal(legacy.H_s,disabled.H_s) && isequal(legacy_image,disabled_image) && ...
    isequal(legacy_k,disabled_k) && isequal(legacy_value,disabled_value) && ...
    isequal(legacy_gradient,disabled_gradient), ...
    'Disabling the response must preserve the original model exactly.');
report.disabled_exact_legacy_equivalence = true;
disabled_diagnostics = acdc_response_diagnostics(currents,disabled,500,30);
assert(isequal(disabled_diagnostics.actual_currents_at_adc_A, ...
    disabled_diagnostics.commanded_currents_at_adc_A) && ...
    isequal(disabled_diagnostics.integrated_currents_to_adc_As, ...
    disabled_diagnostics.commanded_integrated_currents_to_adc_As) && ...
    isequal(disabled_diagnostics.response_image,disabled_diagnostics.instantaneous_image), ...
    'Disabled diagnostics must reproduce DAC commands and instantaneous phase.');

% Default shared tau and a deliberately diverse per-channel case exercise
% both execution paths. Different ADC block sizes must preserve the physics.
report.cases = struct([]);
tau_cases = {tau,linspace(10e-6,160e-6,32).'};
for test = 1:numel(tau_cases)
    dynamic_experiment = experiment;
    dynamic_experiment.current_response = struct('enabled',true,'tau_s',tau_cases{test});
    ctx = prepare_experiment_optimizer(dynamic_experiment,calibration,1);
    [zero_image,zero_k] = forward_model_experiment_acdc(zeros(32),ctx);
    [legacy_zero,legacy_zero_k] = forward_model_experiment_acdc(zeros(32),legacy);
    assert(isequal(zero_image,legacy_zero) && isequal(zero_k,legacy_zero_k), ...
        'Zero currents must preserve the uncorrected baseline.');
    [image,kdata] = forward_model_experiment_acdc(currents,ctx);
    integrated_currents = zeros(32,1024);
    for channel = 1:32
        integrated_currents(channel,:) = currents(channel,:)*ctx.H_response_s(:,:,channel).';
    end
    expected_signal = sum(ctx.base_signal.*exp(-2*pi*1i*(ctx.B*integrated_currents)),1).';
    assert(norm(kdata(:)-expected_signal)/norm(expected_signal)<1e-13, ...
        'The signal does not use the full real-field integral for every channel.');
    [value,gradient] = obj_fun_experiment(x,ctx);
    assert(norm(image(:)-legacy_image(:))>1e-8, 'Finite tau has no simulated phase effect.');
    row = struct('tau_s',ctx.current_response.tau_s, ...
        'directional_gradient_relative_errors',gradient_checks(x,ctx,gradient), ...
        'accelerated_image_relative_error',0,'accelerated_gradient_relative_error',0, ...
        'accelerated_value_relative_error',0);
    for block_size = [17 128 1024]
        runtime = prepare_experiment_runtime(ctx,'Device','cpu','BlockSize',block_size);
        [fast_image,fast_k] = forward_model_experiment_accelerated(currents,runtime);
        [fast_value,fast_gradient] = obj_fun_experiment_accelerated(x,runtime);
        fast_value_only = obj_fun_experiment_accelerated(x,runtime);
        row.accelerated_image_relative_error = max(row.accelerated_image_relative_error, ...
            max(norm(fast_image(:)-image(:))/norm(image(:)), ...
                norm(fast_k(:)-kdata(:))/norm(kdata(:))));
        row.accelerated_gradient_relative_error = max(row.accelerated_gradient_relative_error, ...
            norm(fast_gradient-gradient)/max(norm(gradient),1e-12));
        row.accelerated_value_relative_error = max(row.accelerated_value_relative_error, ...
            abs(fast_value-value)/max(abs(value),1e-12));
        assert(abs(fast_value_only-fast_value)<1e-13*max(1,abs(fast_value)), ...
            'Accelerated objective-only and adjoint calls disagree.');
        assert(max([row.accelerated_image_relative_error, ...
            row.accelerated_gradient_relative_error,row.accelerated_value_relative_error])<1e-10, ...
            'The accelerated CPU path changed the dynamic response model.');
    end
    % Diagnostics must reconstruct a cached runtime in the original ADC order.
    diagnostics = acdc_response_diagnostics(currents,ctx,500,30);
    cached_diagnostics = acdc_response_diagnostics(currents,runtime,500,30);
    assert(isequal(size(diagnostics.actual_currents_at_adc_A),[32 1024]) && ...
        isequal(size(diagnostics.integrated_currents_to_adc_As),[32 1024]), ...
        'Diagnostic currents must follow channel-by-ADC dimensions.');
    assert(norm(diagnostics.response_image(:)-image(:))/norm(image(:))<1e-13 && ...
        abs(diagnostics.response_relative_mse-value)<1e-13 && ...
        norm(diagnostics.instantaneous_image(:)-legacy_image(:))/norm(legacy_image(:))<1e-13, ...
        'Diagnostic images or MSE differ from their forward models.');
    assert(norm(cached_diagnostics.response_image(:)-image(:))/norm(image(:))<1e-13 && ...
        isequal(cached_diagnostics.actual_currents_at_adc_A,diagnostics.actual_currents_at_adc_A), ...
        'Cached-runtime diagnostics changed the current history or ADC order.');
    assert(max(abs(diagnostics.step_reach99_time_s-ctx.current_response.tau_s*log(100)))<1e-18, ...
        'Diagnostic settling times differ from the configured tau values.');
    for channel = 1:32
        expected_current = currents(channel,:)*ctx.R_response(:,:,channel).';
        expected_integral = currents(channel,:)*ctx.H_response_s(:,:,channel).';
        assert(max(abs(diagnostics.actual_currents_at_adc_A(channel,:)-expected_current))<1e-14 && ...
            max(abs(diagnostics.integrated_currents_to_adc_As(channel,:)-expected_integral))<1e-17, ...
            'Diagnostic currents or integrals differ from the response weights.');
    end
    if test==1, report.cases = row; else, report.cases(test) = row; end
end

% Even when command windows last longer than tau, each switch loses a small
% phase area. Evaluating only the current at ADC would miss that difference.
dynamic_experiment.current_response.tau_s = 1e-12;
tiny_ctx = prepare_experiment_optimizer(dynamic_experiment,calibration,1);
tiny_image = forward_model_experiment_acdc(currents,tiny_ctx);
report.tiny_tau_image_relative_error = norm(tiny_image(:)-legacy_image(:))/norm(legacy_image(:));
assert(report.tiny_tau_image_relative_error<1e-8, 'Tiny tau failed the instantaneous image limit.');
bad = experiment; bad.current_response = struct('enabled',true,'tau_s',ones(31,1)*tau);
expect_failure(@() prepare_experiment_optimizer(bad,calibration,1));
bad.current_response.tau_s = zeros(32,1);
expect_failure(@() prepare_experiment_optimizer(bad,calibration,1));
bad.current_response.tau_s = tau;
bad.acquisition.sequence_kind = 'gre';
expect_failure(@() prepare_experiment_optimizer(bad,calibration,1));
report.invalid_inputs_rejected = true;
report.status = 'passed';
fprintf('ACDC response validation: I(133 us)=%.12f; history %.3g A; integral %.3g A*s.\n', ...
    report.step_at_133_us,report.history_absolute_error_A,report.integral_absolute_error_As);
for test = 1:numel(report.cases)
    row = report.cases(test);
    fprintf('Response case %d: adjoint max %.3g; accelerated image %.3g, gradient %.3g.\n', ...
        test,max(row.directional_gradient_relative_errors), ...
        row.accelerated_image_relative_error,row.accelerated_gradient_relative_error);
end
end

function actual = actual_from_recurrence(times,starts,ends,commands,tau)
% Independent piecewise solution retaining the attained current at edges.
edges = unique([0; starts; ends(isfinite(ends))]);
actual = zeros(size(times));
old = 0;
for interval = 1:numel(edges)
    left = edges(interval);
    if interval<numel(edges), right = edges(interval+1); else, right = Inf; end
    command = sum(commands(starts<=left & ends>left));
    selected = times>=left & times<right;
    actual(selected) = command+(old-command)*exp(-(times(selected)-left)/tau);
    if isfinite(right), old = command+(old-command)*exp(-(right-left)/tau); end
end
end

function errors = gradient_checks(x,ctx,gradient)
% Include early/late command states and channels at opposite tau extremes.
directions = randn(numel(x),2);
directions(:,3:4) = 0;
directions(1,3) = 1;
directions(end,4) = 1;
errors = zeros(1,size(directions,2));
for test = 1:size(directions,2)
    direction = directions(:,test)/norm(directions(:,test));
    step = 1e-5;
    numerical = (obj_fun_experiment(x+step*direction,ctx) ...
        -obj_fun_experiment(x-step*direction,ctx))/(2*step);
    analytical = gradient.'*direction;
    errors(test) = abs(numerical-analytical)/max([abs(numerical),abs(analytical),1e-8]);
end
assert(all(errors<2e-6), 'The dynamic objective failed independent adjoint checks.');
end

function expect_failure(call)
failed = false;
try, call(); catch, failed = true; end
assert(failed,'Invalid response metadata was accepted.');
end

function [experiment,calibration] = synthetic_epi()
% Small object support but all 32 channels and 32 acquired EPI lines.
object_size = [8 8];
[X,Y] = meshgrid(((0:7)-3.5)*.2/8);
m0 = zeros(object_size);
points = [10 13 20 27 35 44 49 54 60];
m0(points) = .3+rand(size(points));
b0 = zeros(object_size); b0(points) = 60*randn(size(points));
starts = .001+(0:31)'*180e-6;
ends = [starts(2:end); starts(end)+180e-6];
times = starts+3e-6+(0:31)*5e-6;
times(2:2:end,:) = fliplr(times(2:2:end,:));
[kx,ky] = meshgrid((-16:15)/.2);
experiment = struct('geometry',struct('X_m',X,'Y_m',Y), ...
    'm0_proxy',m0,'field',struct('b0Hz',b0,'mask',m0>0,'simulation_mask',m0>0), ...
    'relaxation',struct('T2star_s',.05), ...
    'acquisition',struct('kx_mInv',kx,'ky_mInv',ky,'time_s',times, ...
        'TE_s',.0038,'line_start_s',starts,'line_end_s',ends, ...
        'line_index',repmat((1:32)',1,32),'sequence_kind','epi'));
calibration = struct('X_m',X,'Y_m',Y,'units','Hz/A', ...
    'acdc_maps',reshape(100*randn(prod(object_size),32),[object_size 32]));
end
