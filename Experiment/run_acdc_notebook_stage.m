function [result_file, status] = run_acdc_notebook_stage(request_file, stage)
%RUN_ACDC_NOTEBOOK_STAGE Keep the forward model and optimization in MATLAB.
%   A notebook writes one MAT file containing a scalar struct named request.
%   Required field: output_dir; forward/optimize also need input_file.
%   The forward stage needs no
%   hardware limits. The optimize stage also needs calibration_file,
%   max_current_A, total_current_limit_A, field_limit_Hz, max_iterations,
%   workers, threads_per_worker, device and scratch_dir. Optional target_mse
%   stops a feasible solution below that relative magnitude MSE ([] disables).
%   Optional initial_currents_A supplies a channel vector or a channels-by-
%   states matrix of initial DAC guesses. Omission gives zero guesses and
%   does not change the physical zero initial current in the response model.
%   The waveform stage samples saved DAC commands without optimization.
%   It requires waveform_currents_A, waveform_line_start_s,
%   waveform_line_end_s and waveform_current_response in the request.
%   Results and status are saved even when a stage fails. A completed solver
%   run is not necessarily converged: inspect the saved diagnostic flags.

stage = validatestring(lower(char(stage)), {'forward', 'optimize', 'waveform'});
loaded = load(request_file, 'request');
assert(isfield(loaded, 'request') && isstruct(loaded.request) && ...
    isscalar(loaded.request), 'Request MAT must contain a scalar struct request.');
request = loaded.request;
assert(isfield(request, 'output_dir'), 'Request needs output_dir.');
output_dir = char(request.output_dir);
if ~isfolder(output_dir), mkdir(output_dir); end
if strcmp(stage, 'forward')
    result_file = fullfile(output_dir, 'forward.mat');
elseif strcmp(stage, 'waveform')
    result_file = fullfile(output_dir, 'waveform_preview.mat');
else
    result_file = fullfile(output_dir, 'optimization_stage.mat');
end
status = struct('stage', stage, 'state', 'running', 'message', '', ...
    'error_identifier', '', 'error_report', '', 'result_file', result_file);
timer = tic;
try
    if ~strcmp(stage, 'waveform')
        assert(isfield(request, 'input_file'), 'Request needs input_file.');
    end
    if strcmp(stage, 'forward')
        loaded = load(char(request.input_file), 'experiment');
        assert(isfield(loaded, 'experiment'), 'Input MAT needs struct experiment.');
        experiment = loaded.experiment;
        m0_proxy = double(experiment.m0_proxy);
        b0Hz = double(experiment.field.b0Hz);
        simulation_mask = logical(experiment.field.simulation_mask) & m0_proxy > 0;
        quality_mask = logical(experiment.field.mask) & simulation_mask;
        m0_proxy(~simulation_mask) = 0;
        b0Hz(~simulation_mask) = 0;
        geometry = experiment.geometry;
        acquisition = experiment.acquisition;
        relaxation = experiment.relaxation;
        [ideal_image, ideal_kdata] = forward_model_experiment(m0_proxy, ...
            zeros(size(b0Hz)), geometry, acquisition, relaxation);
        [measured_image, measured_kdata] = forward_model_experiment(m0_proxy, ...
            b0Hz, geometry, acquisition, relaxation);
        measured_scan = experiment.images.scan;
        % Use the same magnitude-image objective as the existing optimizer.
        relative_mse = sum((abs(measured_image(:))-abs(ideal_image(:))).^2) / ...
            max(sum(abs(ideal_image(:)).^2), eps);
        % A unit-step diagnostic is calculated in MATLAB, independently of
        % optimization. Zero-current baseline images do not depend on tau.
        [current_response, response_preview] = unit_step_preview(experiment);
        save(result_file, 'ideal_image', 'ideal_kdata', 'measured_image', ...
            'measured_kdata', 'measured_scan', 'm0_proxy', 'b0Hz', ...
            'quality_mask', 'simulation_mask', 'geometry', 'acquisition', ...
            'relaxation', 'relative_mse', 'current_response', 'response_preview');
        status.message = 'Zero-B0 and measured-B0 forward models completed.';
    elseif strcmp(stage, 'waveform')
        required = {'waveform_currents_A','waveform_line_start_s', ...
            'waveform_line_end_s','waveform_current_response'};
        assert(all(isfield(request,required)), ...
            'Waveform preview needs DAC commands, command windows and response parameters.');
        current_response = request.waveform_current_response;
        [waveform_actual_currents_A,waveform_commanded_currents_A,waveform_time_s] = ...
            acdc_sample_current_waveform(request.waveform_currents_A, ...
                request.waveform_line_start_s,request.waveform_line_end_s,[],current_response);
        save(result_file,'waveform_time_s','waveform_actual_currents_A', ...
            'waveform_commanded_currents_A','current_response');
        status.message = 'Complete predicted current waveform sampled; no optimization was run.';
    else
        required = {'calibration_file', 'max_current_A', 'total_current_limit_A', ...
            'field_limit_Hz', 'max_iterations', 'workers', 'threads_per_worker', ...
            'device', 'scratch_dir'};
        assert(all(isfield(request, required)), ...
            'Optimization request is missing explicitly chosen solver parameters.');
        optimization_dir = fullfile(output_dir, 'optimization');
        target_mse = [];
        if isfield(request, 'target_mse'), target_mse = request.target_mse; end
        initial_currents_A = [];
        if isfield(request, 'initial_currents_A'), initial_currents_A = request.initial_currents_A; end
        [run_dir, summary] = run_experiment_fast_parallel( ...
            char(request.calibration_file), request.max_current_A, request.workers, ...
            'InputFile', char(request.input_file), 'OutputDir', optimization_dir, ...
            'FieldLimits', request.field_limit_Hz, 'MaxIterations', request.max_iterations, ...
            'TargetMSE', target_mse, ...
            'InitialCurrents_A', initial_currents_A, ...
            'TotalCurrentLimit_A', request.total_current_limit_A, ...
            'ThreadsPerWorker', request.threads_per_worker, 'Device', char(request.device), ...
            'ScratchDir', char(request.scratch_dir), 'MakeFigures', false);
        status.run_dir = run_dir;
        status.all_converged = summary.all_converged;
        status.all_accepted = summary.all_accepted;
        status.all_current_limits_met = summary.all_current_limits_met;
        status.all_field_limits_met = summary.all_field_limits_met;
        status.message = 'Solver completed. Check target, convergence and constraint flags.';
        save(result_file, 'run_dir', 'summary');
    end
    status.state = 'complete';
catch err
    status.state = 'failed';
    status.message = err.message;
    status.error_identifier = err.identifier;
    status.error_report = getReport(err,'extended','hyperlinks','off');
    % Existing optimization outputs retain any successfully completed limits.
    fprintf(2, 'Notebook %s stage failed:\n%s\n', stage, status.error_report);
end
status.elapsed_seconds = toc(timer);
if isfile(result_file)
    save(result_file, 'status', '-append');
else
    save(result_file, 'status');
end
fprintf('Notebook %s stage: %s. Results: %s\n', stage, status.state, result_file);
end

function [response, preview] = unit_step_preview(experiment)
response = struct('enabled',false,'tau_s',repmat(133e-6/log(100),32,1));
if isfield(experiment,'current_response')
    config = experiment.current_response;
    assert(isstruct(config) && isscalar(config), ...
        'experiment.current_response must be a scalar struct.');
    if isfield(config,'enabled')
        flag = config.enabled;
        assert((islogical(flag) || isnumeric(flag)) && isreal(flag) && ...
            isscalar(flag) && any(double(flag)==[0 1]), ...
            'current_response.enabled must be true or false.');
        response.enabled = logical(flag);
    end
    if isfield(config,'tau_s')
        tau = double(config.tau_s(:));
        if isscalar(tau), tau = repmat(tau,32,1); end
        response.tau_s = tau;
    end
end
validateattributes(response.tau_s,{'numeric'}, ...
    {'real','finite','positive','size',[32 1]});
time_s = linspace(0,6*max(response.tau_s),401).';
[H, R] = acdc_current_response(0,Inf,time_s,response.tau_s);
actual_A = reshape(R,numel(time_s),32);
integral_As = reshape(H,numel(time_s),32);
if ~response.enabled
    actual_A(:) = 1;
    integral_As = repmat(time_s,1,32);
end
preview = struct('time_s',time_s,'actual_A',actual_A, ...
    'integrated_As',integral_As,'command_A',1, ...
    't99_s',log(100)*response.tau_s,'initial_current_A',0);
end
