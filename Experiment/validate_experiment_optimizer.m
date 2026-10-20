function report = validate_experiment_optimizer(run_pool)
%VALIDATE_EXPERIMENT_OPTIMIZER Synthetic numerical tests, never real currents.
%   Optional RUN_POOL=true tests two short SYNTHETIC jobs at zero iterations,
%   saved only under a temporary folder deleted afterward. These synthetic
%   maps are test fixtures and MUST NOT be used for the real experiment.
if nargin<1, run_pool = false; end
rng(27);
size_object = [128 128];
[X,Y] = meshgrid(((0:127)-64)*0.2/128);
experiment = struct;
experiment.geometry = struct('X_m',X,'Y_m',Y);
experiment.m0_proxy = zeros(size_object);
points = [sub2ind(size_object,42,58),sub2ind(size_object,75,80),sub2ind(size_object,91,45)];
experiment.m0_proxy(points) = [1 .7 .4];
experiment.field = struct('b0Hz',zeros(size_object),'mask',experiment.m0_proxy>0, ...
    'simulation_mask',experiment.m0_proxy>0);
experiment.field.b0Hz(points) = [20 -15 7];
experiment.relaxation.T2star_s = Inf;
line_start = .005+(0:31)'*.00054;
time = line_start+.0001+(0:31)*.000011;
time(2:2:end,:) = fliplr(time(2:2:end,:));
[kx,ky] = meshgrid((-16:15)/.2);
experiment.acquisition = struct('kx_mInv',kx,'ky_mInv',ky,'time_s',time, ...
    'TE_s',.0127645,'line_start_s',line_start,'line_index',repmat((1:32)',1,32));
calibration = struct('X_m',X,'Y_m',Y,'units','Hz/A','acdc_maps',zeros([size_object 32]));
for channel = 1:32
    map = zeros(size_object);
    map(points) = 100*randn(1,numel(points));
    calibration.acdc_maps(:,:,channel) = map;
end
ctx = prepare_experiment_optimizer(experiment,calibration,.2);
[zero_image,zero_k] = forward_model_experiment_acdc(zeros(32),ctx);
[static_image,static_k] = forward_model_experiment(experiment.m0_proxy, ...
    experiment.field.b0Hz,experiment.geometry,experiment.acquisition,experiment.relaxation);
report.zero_current_relative_error = norm(zero_image(:)-static_image(:))/norm(static_image(:));
assert(report.zero_current_relative_error<1e-13 && norm(zero_k(:)-static_k(:))/norm(static_k(:))<1e-13, ...
    'Zero currents must reproduce the static experimental forward model.');
[ideal_image,~] = forward_model_experiment(experiment.m0_proxy,zeros(size_object), ...
    experiment.geometry,experiment.acquisition,experiment.relaxation);
assert(norm(ctx.reference(:)-ideal_image(:))/norm(ideal_image(:))<1e-13, 'Wrong ideal reference.');

% All states at the same current: shim phase begins at the first IMAGING
% block, while original B0 phase has accumulated since the RF center.
constant_current = .003*randn(32,1);
currents = repmat(constant_current,1,32);
[~,k_constant] = forward_model_experiment_acdc(currents,ctx);
field = ctx.B*constant_current;
expected_signal = sum(ctx.base_signal.*exp(-2*pi*1i*field ...
    .*max(0,experiment.acquisition.time_s(:).'-line_start(1))),1).';
report.accumulated_phase_relative_error = norm(k_constant(:)-expected_signal)/norm(expected_signal);
assert(report.accumulated_phase_relative_error<1e-13, 'Incorrect state-duration integration.');
assert(all(ctx.H_s(:,end)==max(0,time(:)-line_start(end))), 'The last line state is mis-timed.');

% Different channel/state directions verify adjoint sign, FFT scaling,
% accumulation into EARLIER states and nominal ky/kx sample ordering.
x = .001*randn(1024,1);
[value,gradient] = obj_fun_experiment(x,ctx);
errors = zeros(1,4);
for test = 1:4
    direction = randn(1024,1); direction = direction/norm(direction);
    step = 1e-6;
    numerical = (obj_fun_experiment(x+step*direction,ctx) ...
        -obj_fun_experiment(x-step*direction,ctx))/(2*step);
    analytical = gradient.'*direction;
    errors(test) = abs(numerical-analytical)/max([abs(numerical),abs(analytical),1e-8]);
end
report.directional_gradient_relative_errors = errors;
assert(all(errors<1e-6), 'Magnitude objective gradient failed directional checks.');
assert(isfinite(value) && all(isfinite(gradient)), 'Nonfinite objective/gradient.');

% A missing physical calibration cannot start a pool or create outputs.
tmp = tempname;
mkdir(tmp);
cleanup = onCleanup(@() rmdir(tmp,'s')); %#ok<NASGU>
output_dir = fullfile(tmp,'must_not_exist');
try
    run_experiment_fast_parallel(fullfile(tmp,'missing.mat'),.2,2,'OutputDir',output_dir);
    error('validate_experiment_optimizer:ExpectedFailure','Missing calibration was accepted.');
catch err
    assert(strcmp(err.identifier,'run_experiment_fast_parallel:MissingCalibration'), ...
        'Missing calibration must fail with its explicit error.');
end
assert(~isfolder(output_dir), 'Missing calibration created output files.');
bad = calibration; bad.X_m = bad.X_m+.001;
expect_preparation_failure(experiment,bad,.2);
bad = calibration; bad.units = 'Hz';
expect_preparation_failure(experiment,bad,.2);
expect_preparation_failure(experiment,calibration,[]);
report.missing_basis_and_metadata_rejected = true;
report.pool_tested = logical(run_pool);
if run_pool
    assert(isempty(gcp('nocreate')), 'Run validation in a new MATLAB session.');
    input_file = fullfile(tmp,'synthetic_input.mat');
    calibration_file = fullfile(tmp,'SYNTHETIC_TEST_ONLY.mat');
    save(input_file,'experiment');
    save(calibration_file,'-struct','calibration');
    previous_threads = maxNumCompThreads;
    [folder,summary] = run_experiment_fast_parallel(calibration_file,.2,2, ...
        'InputFile',input_file,'OutputDir',fullfile(tmp,'synthetic_results'), ...
        'FieldLimits',[100 200],'MaxIterations',0,'ThreadsPerWorker',2);
    assert(strcmp(summary.status,'complete') && numel(summary.completed_limits_Hz)==2, ...
        'Synthetic jobs did not complete.');
    assert(isempty(gcp('nocreate')) && maxNumCompThreads==previous_threads, ...
        'Owned pool/thread settings were not cleaned up.');
    result = load(fullfile(folder,'solution_100Hz.mat'));
    assert(result.worker_threads==2 && isequal(size(result.currents_A),[32 32]) && ...
        norm(result.image(:)-zero_image(:))/norm(zero_image(:))<1e-13, ...
        'Zero-iteration synthetic solve changed the starting state.');
    report.pool_cleanup_verified = true;

    % Explicitly verify that a pre-existing pool is refused and untouched.
    cluster = parcluster('Processes'); cluster.NumWorkers = 1; cluster.NumThreads = 1;
    existing = parpool(cluster,1);
    pool_cleanup = onCleanup(@() delete(existing));
    refused_output = fullfile(tmp,'existing_pool_must_not_write');
    failed = false;
    try
        run_experiment_fast_parallel(calibration_file,.2,2, ...
            'InputFile',input_file,'OutputDir',refused_output,'MaxIterations',0);
    catch err
        failed = contains(err.message,'pool already exists');
    end
    assert(failed && isequal(gcp('nocreate'),existing) && ~isfolder(refused_output), ...
        'Existing pool must be refused intact without output creation.');
    clear pool_cleanup;
    report.existing_pool_refusal_verified = true;
end
fprintf('Synthetic optimizer validation: static %.3g; phase %.3g; gradient max %.3g; pool %d.\n', ...
    report.zero_current_relative_error,report.accumulated_phase_relative_error,max(errors),run_pool);
end

function expect_preparation_failure(experiment,calibration,Imax_A)
failed = false;
try, prepare_experiment_optimizer(experiment,calibration,Imax_A);
catch, failed = true; end
assert(failed,'Invalid geometry/units/current limits were accepted.');
end
