function [run_dir, summary] = run_experiment_fast_parallel(calibration_file, Imax_A, nWorkers, varargin)
%RUN_EXPERIMENT_FAST_PARALLEL Optional optimizer for CALIBRATED experimental ACDC.
%   run_experiment_fast_parallel('calibration.mat',actual_Imax_A,2)
%   requires measured acdc_maps [object rows, object columns, 32] in Hz/A,
%   X_m, Y_m, units='Hz/A', on the input reference image's grid.
%   Imax_A must be supplied explicitly from the actual hardware (A, scalar
%   or 32 channel limits); the old subject-006 basis is never loaded.
%   Options: InputFile (prepared/experiment_input.mat), MaxIterations (600),
%   InitialCurrents_A ([] gives zero DAC guesses; channel vector repeats over
%   all states, or supply a channels-by-states matrix). This solver seed does
%   not change the physical initial current, which remains zero.
%   TargetMSE ([] disables early stopping on feasible relative magnitude MSE),
%   FieldLimits ([100 200 300 500 1000] Hz), ThreadsPerWorker (1, allowed 1..6),
%   TotalCurrentLimit_A (30), Device ('cpu' or 'gpu'), BlockSize ([]),
%   SubproblemAlgorithm ('cg' or 'factorization'), PruneFieldConstraints (true),
%   CalibrationMode ('registered' or explicit 'virtual'), ScratchDir ('/scratch'),
%   MakeFigures (false), OutputDir (new folder only). At most five independent
%   jobs run; workers*threads+one client thread must remain below 32.
%   GPU uses one worker, one resident cache; fmincon/constraints stay on CPU.
%   experiment.current_response enables an approximate first-order EPI channel
%   response with scalar or per-channel tau_s. Omitted/disabled settings retain
%   instantaneous changes. Currents initially equal zero; their complete
%   history is integrated into RF-to-ADC phase. Spoiled GRE retains its existing
%   instantaneous model. No measured RF/receiver-sensitivity model is available.
%   Existing optimization bounds constrain DAC commands. Predicted real current
%   and field limits are checked separately at ADC samples in saved diagnostics.
%   Do not interpret this as a
%   hardware programming procedure; additional real hardware limits matter.

if nargin<1 || isempty(calibration_file) || ~isfile(calibration_file)
    error('run_experiment_fast_parallel:MissingCalibration', ...
        'Supply a measured experimental ACDC calibration MAT (acdc_maps in Hz/A, X_m, Y_m, units).');
end
if nargin<2 || isempty(Imax_A)
    error('run_experiment_fast_parallel:MissingCurrentLimit', ...
        'Supply actual allowed channel currents Imax_A explicitly in amperes.');
end
if nargin<3 || isempty(nWorkers), nWorkers = 2; end
validateattributes(nWorkers,{'numeric'},{'real','scalar','finite','integer','>=',1,'<',32});
root = fileparts(mfilename('fullpath'));
parser = inputParser;
parser.PartialMatching = false;
addParameter(parser,'InputFile',fullfile(root,'prepared','experiment_input.mat'),@text_scalar);
addParameter(parser,'OutputDir','',@text_scalar);
addParameter(parser,'ThreadsPerWorker',1,@(v) isnumeric(v) && isreal(v) && ...
    isscalar(v) && isfinite(v) && v==fix(v) && v>=1 && v<=6);
addParameter(parser,'MaxIterations',600,@(v) isnumeric(v) && isreal(v) && ...
    isscalar(v) && isfinite(v) && v==fix(v) && v>=0);
addParameter(parser,'TargetMSE',[],@(v) isempty(v) || (isnumeric(v) && isreal(v) && ...
    isscalar(v) && isfinite(v) && v>0));
addParameter(parser,'InitialCurrents_A',[],@(v) isnumeric(v) && isreal(v) && ...
    ismatrix(v) && all(isfinite(v(:))));
addParameter(parser,'FieldLimits',[100 200 300 500 1000],@(v) isnumeric(v) && ...
    isreal(v) && isvector(v) && ~isempty(v) && all(isfinite(v)) && all(v>0));
addParameter(parser,'MakeFigures',false,@(v) islogical(v) && isscalar(v));
addParameter(parser,'TotalCurrentLimit_A',30,@(v) isnumeric(v) && isreal(v) && ...
    isscalar(v) && isfinite(v) && v>1e-4);
addParameter(parser,'Device','cpu',@text_scalar);
addParameter(parser,'BlockSize',[],@(v) isempty(v) || (isnumeric(v) && isreal(v) && ...
    isscalar(v) && isfinite(v) && v==fix(v) && v>=1));
addParameter(parser,'CalibrationMode','registered',@text_scalar);
addParameter(parser,'ScratchDir','/scratch',@text_scalar);
addParameter(parser,'SubproblemAlgorithm','cg',@text_scalar);
addParameter(parser,'PruneFieldConstraints',true,@(v) islogical(v) && isscalar(v));
parse(parser,varargin{:});
config = parser.Results;
config.Device = validatestring(lower(char(config.Device)),{'cpu','gpu'});
config.CalibrationMode = validatestring(lower(char(config.CalibrationMode)),{'registered','virtual'});
config.SubproblemAlgorithm = validatestring(lower(char(config.SubproblemAlgorithm)),{'cg','factorization'});
limits = double(config.FieldLimits(:).');
assert(numel(limits)<=5 && numel(unique(limits))==numel(limits), ...
    'Choose at most five distinct positive field limits in Hz.');
workers = min(double(nWorkers),numel(limits));
assert(~strcmp(config.Device,'gpu') || workers==1, ...
    'GPU execution uses one worker for the single GPU; request nWorkers=1.');
assert(workers*config.ThreadsPerWorker+1<32, 'workers*ThreadsPerWorker+1 must be below 32.');
assert(isfolder(config.ScratchDir),'ScratchDir must exist and be writable.');
run_dir = char(config.OutputDir);
assert(isempty(run_dir) || (~isfolder(run_dir) && ~isfile(run_dir)), 'OutputDir must not already exist.');

% All missing-basis/unit/geometry/current/sequence checks precede ANY output
% creation or pool startup. Passing calibration alone cannot run this code.
calibration = load(calibration_file);
if strcmp(config.CalibrationMode,'registered') && ...
        isfield(calibration,'registration_confirmed') && ~logical(calibration.registration_confirmed)
    error('run_experiment_fast_parallel:UnregisteredCalibration', ...
        ['These maps describe a virtual coil placement, not a confirmed acquisition registration. ' ...
         'For the phantom experiment without ACDC, explicitly use CalibrationMode=''virtual''.']);
end
loaded = load(config.InputFile,'experiment');
assert(isfield(loaded,'experiment'), 'InputFile must contain struct experiment.');
experiment = loaded.experiment;
ctx = prepare_experiment_optimizer(experiment,calibration,Imax_A);
[config.InitialCurrents_A,initialization_mode] = prepare_experiment_initial_currents( ...
    config.InitialCurrents_A,ctx,config.TotalCurrentLimit_A,limits);
assert(license('test','Distrib_Computing_Toolbox') && license('test','Optimization_Toolbox'), ...
    'Parallel Computing Toolbox and Optimization Toolbox are required.');
assert(isempty(gcp('nocreate')), 'A pool already exists; use a new MATLAB session. Existing pools are left intact.');
[initial_image,initial_kdata] = forward_model_experiment_acdc(zeros(ctx.nc,ctx.nstates),ctx);
if isempty(run_dir)
    results_dir = fullfile(root,'results');
    if ~isfolder(results_dir), mkdir(results_dir); end
    [~,token] = fileparts(tempname(results_dir));
    run_dir = fullfile(results_dir,['optimization_' char(datetime('now','Format','yyyyMMdd_HHmmss')) '_' token]);
end
[ok,message] = mkdir(run_dir);
assert(ok,'Cannot create output folder: %s',message);
[ok,attributes] = fileattrib(run_dir);
assert(ok,'Cannot resolve output folder.');
run_dir = attributes.Name;
fid = fopen(fullfile(run_dir,'optimization_log.txt'),'w');
assert(fid>=0,'Cannot write optimization log.');
resources = containers.Map('KeyType','char','ValueType','any');
resources('fid') = fid;
resources('previous_threads') = maxNumCompThreads;
resources('pool') = [];
resources('constant') = [];
resources('listener') = [];
resources('futures') = [];
resources('job_storage') = '';
cleanup = onCleanup(@() cleanup_resources(resources));
maxNumCompThreads(1);
timer = tic;
summary = struct('status','running','field_limits_Hz',limits,'nworkers',workers, ...
    'target_mse',double(config.TargetMSE), ...
    'threads_per_worker',config.ThreadsPerWorker,'completed_limits_Hz',[], ...
    'files',{cell(size(limits))},'optimization_info',{cell(size(limits))}, ...
    'total_seconds',0,'calibration_file',char(calibration_file),'Imax_A',ctx.Imax_A, ...
    'TotalCurrentLimit_A',config.TotalCurrentLimit_A,'current_sum_kind','Sum of absolute currents per imaging state', ...
    'current_sum_numerical_margin_A',1e-4,'current_smoothing_A',1e-6, ...
    'device',config.Device,'precision','double','calibration_mode',config.CalibrationMode, ...
    'subproblem_algorithm',config.SubproblemAlgorithm, ...
    'registration_confirmed',ctx.registration_confirmed, ...
    'calibration_units','Hz/A','current_states',[ctx.nc ctx.nstates], ...
    'sequence_kind',ctx.sequence_kind,'object_size',ctx.object_size,'image_size',ctx.image_size, ...
    'navigator_currents_A',0,'baseline_B0_measured',false, ...
    'target_kind','Ideal zero-B0 twin, not measured baseline field', ...
    'field_constraint_kind','Generated ACDC field on reliable measured B0/object voxels', ...
    'unmeasured_field_extension',any(ctx.support(:) & ~ctx.quality_mask(:)), ...
    'receiver_model','One uniform receive sensitivity', ...
    'initialization_mode',initialization_mode, ...
    'initial_currents_A',config.InitialCurrents_A, ...
    'current_response',ctx.current_response, ...
    'switching_model','Instantaneous currents; no measured RF model');
summary.field_constraint_kind = 'Generated ACDC field from DAC commands on reliable measured B0/object voxels';
if ctx.current_response.enabled
    summary.switching_model = 'First-order channel response; zero initial current; RF-to-ADC phase integrated analytically';
end
try
    log_message(fid,sprintf(['Starting optimization on %s: %d worker(s), ' ...
        '%d current states, at most %d iterations per field limit.\n'], ...
        config.Device,workers,ctx.nstates,config.MaxIterations));
    log_message(fid,sprintf('Image solver seed: %s; maximum absolute DAC command %.9g A.\n', ...
        initialization_mode,max(abs(config.InitialCurrents_A),[],'all')));
    if ctx.current_response.enabled
        log_message(fid,sprintf('First-order EPI response enabled; tau range %.9g to %.9g us; initial current 0 A.\n', ...
            min(ctx.current_response.tau_s)*1e6,max(ctx.current_response.tau_s)*1e6));
    else
        log_message(fid,'First-order response disabled; instantaneous current transitions.\n');
    end
    if ~isempty(config.TargetMSE)
        log_message(fid,sprintf('Stop when relative magnitude MSE < %.9g and current/field bounds are met.\n', ...
            config.TargetMSE));
    end
    baseline = struct('reference',ctx.reference,'initial_image',initial_image, ...
        'initial_kdata',initial_kdata,'m0_proxy',ctx.m0_proxy,'b0Hz',ctx.b0Hz, ...
        'quality_mask',ctx.quality_mask,'simulation_mask',ctx.support, ...
        'calibration_valid_mask',ctx.calibration_valid_mask, ...
        'acquisition',experiment.acquisition,'geometry',experiment.geometry, ...
        'relaxation',experiment.relaxation,'current_response',ctx.current_response);
    if isfield(calibration,'metadata')
        baseline.calibration_metadata = calibration.metadata;
        summary.calibration_metadata = calibration.metadata;
    end
    if isfield(experiment,'provenance'), baseline.provenance = experiment.provenance; end
    if isfield(experiment,'assumptions'), baseline.assumptions = experiment.assumptions; end
    save(fullfile(run_dir,'baseline.mat'),'-struct','baseline');
    cluster = parcluster('Processes');
    cluster.NumWorkers = workers;
    cluster.NumThreads = config.ThreadsPerWorker;
    job_storage = tempname(char(config.ScratchDir));
    [ok,message] = mkdir(job_storage);
    assert(ok,'Cannot create local pool storage: %s',message);
    resources('job_storage') = job_storage;
    cluster.JobStorageLocation = job_storage;
    summary.job_storage_location = job_storage;
    pool = parpool(cluster,workers);
    resources('pool') = pool;
    fetchOutputs(parfevalOnAll(pool,@maxNumCompThreads,0,config.ThreadsPerWorker));
    worker_threads = fetchOutputs(parfevalOnAll(pool,@maxNumCompThreads,1));
    if iscell(worker_threads), worker_threads = cell2mat(worker_threads); end
    assert(numel(worker_threads)==workers && all(worker_threads(:)==config.ThreadsPerWorker), ...
        'Worker thread limit could not be enforced.');
    % Prepare once on each worker, including one persistent GPU transfer.
    ctx_constant = parallel.pool.Constant(@() prepare_experiment_runtime( ...
        ctx,'Device',config.Device,'BlockSize',config.BlockSize));
    resources('constant') = ctx_constant;
    queue = parallel.pool.DataQueue;
    resources('listener') = afterEach(queue,@(data) report_progress(data,fid,config.MaxIterations));
    futures = parallel.FevalFuture.empty(0,1);
    for ii = 1:numel(limits)
        futures(ii,1) = parfeval(pool,@run_experiment_fast_parallel_worker,1, ...
            ctx_constant,limits(ii),config.MaxIterations,config.ThreadsPerWorker,queue,config);
        resources('futures') = futures;
    end
    for completed = 1:numel(limits)
        [ii,result] = fetchNext(futures);
        filename = fullfile(run_dir,sprintf('solution_%gHz.mat',limits(ii)));
        saved = rmfield(result,'solver_log');
        saved.reference = ctx.reference;
        saved.line_start_s = ctx.line_start_s;
        saved.line_end_s = ctx.line_end_s;
        saved.sequence_kind = ctx.sequence_kind;
        saved.rf_time_s = ctx.rf_time_s;
        saved.phase_offset_rad = ctx.phase_offset_rad;
        saved.object_size = ctx.object_size;
        saved.image_size = ctx.image_size;
        saved.Imax_A = ctx.Imax_A;
        saved.calibration_file = char(calibration_file);
        saved.calibration_mode = config.CalibrationMode;
        saved.registration_confirmed = ctx.registration_confirmed;
        saved.calibration_valid_mask = ctx.calibration_valid_mask;
        save(filename,'-struct','saved');
        write_text(fullfile(run_dir,sprintf('solver_%gHz.txt',limits(ii))),result.solver_log);
        summary.completed_limits_Hz(end+1) = limits(ii);
        summary.files{ii} = filename;
        summary.optimization_info{ii} = result.optimization_info;
        summary.total_seconds = toc(timer);
        save(fullfile(run_dir,'summary.mat'),'summary','config');
        if config.MakeFigures, save_figure(result,baseline,run_dir); end
        info = result.optimization_info;
        log_message(fid,sprintf(['[%g Hz] Finished after %d iterations and %d evaluations; ' ...
            'MSE %.9g; exitflag %d; converged %d; accepted %d; current limits met %d; ' ...
            'field limit met %d; elapsed %.1f s.\n'],limits(ii), ...
            info.output.iterations,info.output.funcCount,info.fval,info.exitflag, ...
            info.converged,info.accepted,info.current_limits_met,info.field_limits_met,result.elapsed_seconds));
        log_message(fid,sprintf('[%g Hz] Stopping reason: %s\n',limits(ii),strtrim(info.stopping_reason)));
        diag = result.response_diagnostics;
        log_message(fid,sprintf(['[%g Hz] Predicted real-current ADC checks: maximum generated field %.9g Hz; ' ...
            'maximum absolute current sum %.9g A; field limits met %d; current limits met %d.\n'], ...
            limits(ii),diag.maximum_actual_quality_field_Hz,diag.maximum_actual_current_sum_at_adc_A, ...
            info.actual_adc_field_limits_met,info.actual_adc_current_limits_met));
        log_message(fid,sprintf('[%g Hz] Solver message: %s\n',limits(ii),strtrim(info.output.message)));
        log_message(fid,sprintf('[%g Hz] Saved result: %s\n',limits(ii),filename));
    end
    summary.status = 'complete';
    summary.all_converged = all(cellfun(@(v) v.converged,summary.optimization_info));
    summary.all_accepted = all(cellfun(@(v) v.accepted,summary.optimization_info));
    summary.all_current_limits_met = all(cellfun(@(v) v.current_limits_met,summary.optimization_info));
    summary.all_field_limits_met = all(cellfun(@(v) v.field_limits_met,summary.optimization_info));
    summary.total_seconds = toc(timer);
    save(fullfile(run_dir,'summary.mat'),'summary','config');
    fprintf('Experimental optimization saved: %s. Inspect acceptance and constraint flags.\n',run_dir);
catch err
    summary.status = 'failed';
    summary.total_seconds = toc(timer);
    summary.error_identifier = err.identifier;
    summary.error_message = err.message;
    % A parallel worker/fmincon exception often wraps its actual cause.
    % Preserve the full report instead of only "function evaluation error".
    summary.error_report = getReport(err,'extended','hyperlinks','off');
    log_message(fid,sprintf('Optimization failed:\n%s\n',summary.error_report));
    try, save(fullfile(run_dir,'summary.mat'),'summary','config'); catch, end
    rethrow(err);
end
clear cleanup;
end

function valid = text_scalar(value)
valid = (ischar(value) && (isrow(value) || isempty(value))) || (isstring(value) && isscalar(value));
end

function report_progress(data,fid,cap)
if strcmp(data.state,'init') || strcmp(data.state,'iter')
    message = sprintf(['[%g Hz] Iteration %d/%d (%s); evaluations %g; ' ...
        'MSE %.9g; optimality %.3g; violation %.3g; elapsed %.1f s.\n'], ...
        data.limit_Hz,data.iteration,cap,data.state,data.function_evaluations, ...
        data.fval,data.firstorderopt,data.constrviolation,data.elapsed_seconds);
    log_message(fid,message);
end
end

function log_message(fid,message)
fprintf('%s',message);
fprintf(fid,'%s',message);
end

function cleanup_resources(resources)
futures = resources('futures');
if ~isempty(futures), try, cancel(futures); catch, end, end
listener = resources('listener');
if ~isempty(listener), try, delete(listener); catch, end, end
pool = resources('pool');
if ~isempty(pool), try, delete(pool); catch, end, end
constant = resources('constant');
if ~isempty(constant), try, delete(constant); catch, end, end
job_storage = resources('job_storage');
if ~isempty(job_storage) && isfolder(job_storage), try, rmdir(job_storage,'s'); catch, end, end
maxNumCompThreads(resources('previous_threads'));
fid = resources('fid');
if fid>=0, fclose(fid); end
end

function write_text(filename,value)
fid = fopen(filename,'w');
assert(fid>=0,'Cannot write solver log.');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s',value);
end

function save_figure(result,baseline,run_dir)
fig = figure('Visible','off','Position',[100 100 1200 400]);
cleanup = onCleanup(@() close(fig)); %#ok<NASGU>
panels = {abs(baseline.reference),abs(baseline.initial_image),abs(result.image)};
titles = {'Ideal zero-B0 target (not measured)','Measured total-B0 twin, zero ACDC', ...
    sprintf('Simulated dynamic ACDC, limit %g Hz',result.field_limit_Hz)};
upper = max(cellfun(@(v) max(v(:)),panels));
for ii = 1:3
    subplot(1,3,ii); imagesc(panels{ii}); axis image off; colorbar;
    clim([0 upper]); title(titles{ii},'Interpreter','none');
end
colormap(fig,'gray');
savefig(fig,fullfile(run_dir,sprintf('images_%gHz.fig',result.field_limit_Hz)));
exportgraphics(fig,fullfile(run_dir,sprintf('images_%gHz.png',result.field_limit_Hz)),'Resolution',150);
end
