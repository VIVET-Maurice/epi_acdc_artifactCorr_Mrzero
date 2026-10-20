function report = validate_acdc_notebook_stage(input_file, calibration_file, Imax_A, total_limit_A)
%VALIDATE_ACDC_NOTEBOOK_STAGE Verify MAT exchange and saved solver diagnostics.
%   Uses an existing prepared acquisition and its measured calibration.
%   Only a temporary output folder is written. No hardware is contacted.
validateattributes(Imax_A, {'numeric'}, {'real', 'finite', 'positive', 'vector'});
validateattributes(total_limit_A, {'numeric'}, {'real', 'finite', 'positive', 'scalar'});
folder = tempname;
mkdir(folder);
cleanup = onCleanup(@() rmdir(folder, 's')); %#ok<NASGU>
request = struct('input_file', char(input_file), 'output_dir', folder);
request_file = fullfile(folder, 'request.mat');
save(request_file, 'request');
[result_file, status] = run_acdc_notebook_stage(request_file, 'forward');
assert(strcmp(status.state, 'complete'), 'Forward stage failed: %s', status.message);
result = load(result_file);
assert(all(isfinite(result.ideal_image(:))) && all(isfinite(result.measured_image(:))), ...
    'Forward stage saved nonfinite predictions.');
assert(isequal(size(result.ideal_image), size(result.measured_scan)), ...
    'Predictions and measured scan have different image grids.');

loaded = load(input_file, 'experiment');
calibration = load(calibration_file);
ctx = prepare_experiment_optimizer(loaded.experiment, calibration, Imax_A);
[image, kdata] = forward_model_experiment_acdc(zeros(ctx.nc, ctx.nstates), ctx);
report.forward_relative_error = norm(image(:)-result.measured_image(:)) / max(norm(image(:)), eps);
report.kdata_relative_error = norm(kdata(:)-result.measured_kdata(:)) / max(norm(kdata(:)), eps);
assert(max(report.forward_relative_error, report.kdata_relative_error) < 1e-12, ...
    'Notebook forward results differ from the zero-current ACDC model.');
initial_objective = obj_fun_experiment(zeros(ctx.nc*ctx.nstates, 1), ctx);
report.objective_error = abs(initial_objective-result.relative_mse);
assert(report.objective_error < 1e-12, ...
    'Forward-stage magnitude MSE differs from the optimizer objective.');

% At zero iterations, history must still describe the initial solver state.
config = struct('Device', 'cpu', 'TotalCurrentLimit_A', total_limit_A);
solution = run_experiment_fast_parallel_worker(ctx, 100, 0, 1, [], config);
history = solution.history;
assert(~isempty(history.iteration) && history.iteration(1) == 0 && ...
    isfinite(history.fval(1)), 'Initial solver diagnostics are missing.');
assert(all(diff(history.iteration) >= 0) && ...
    numel(history.fval) == numel(history.firstorderopt) && ...
    numel(history.fval) == numel(history.constrviolation) && ...
    numel(history.fval) == numel(history.function_evaluations), ...
    'Saved solver diagnostic arrays have inconsistent lengths.');
assert(all(isfinite(history.function_evaluations)) && ...
    all(history.function_evaluations >= 1) && ...
    all(diff(history.function_evaluations) >= 0) && ...
    history.function_evaluations(end) <= solution.optimization_info.output.funcCount, ...
    'Saved function-evaluation counts do not match the solver diagnostics.');
report.history_samples = numel(history.iteration);

% Failure must remain inspectable in the notebook instead of losing status.
request.input_file = fullfile(folder, 'missing_input.mat');
save(request_file, 'request');
[result_file, status] = run_acdc_notebook_stage(request_file, 'forward');
assert(strcmp(status.state, 'failed') && ~isempty(status.message), ...
    'Missing input did not return a readable failure status.');
assert(contains(status.error_report, 'missing_input.mat') && ...
    contains(status.error_report, 'run_acdc_notebook_stage'), ...
    'Failure report lost the original error or its MATLAB stack.');
saved = load(result_file, 'status');
assert(strcmp(saved.status.state, 'failed') && ...
    strcmp(saved.status.error_report, status.error_report), ...
    'Failed stage status and full error report were not saved.');
report.failure_status_saved = true;
fprintf('Notebook bridge validation: image %.3g; k-space %.3g; %d initial history samples; failure saved.\n', ...
    report.forward_relative_error, report.kdata_relative_error, report.history_samples);
end
