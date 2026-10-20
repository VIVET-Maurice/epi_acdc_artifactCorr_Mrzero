function report = validate_experiment_initial_currents(run_bridge)
%VALIDATE_EXPERIMENT_INITIAL_CURRENTS Check zero and static solver seeds.
%   Synthetic fields/images only; no hardware. Optional true additionally
%   exercises MAT exchange and a one-worker pool in a temporary directory.
if nargin<1, run_bridge = false; end
validateattributes(run_bridge,{'logical'},{'scalar'});

ctx = synthetic_context(false);
config = struct('Device','cpu','TotalCurrentLimit_A',30);
zero = run_experiment_fast_parallel_worker(ctx,100,0,1,[],config);
assert(isequal(zero.initial_currents_A,zeros(ctx.nc,ctx.nstates)) && ...
    isequal(zero.currents_A,zero.initial_currents_A), ...
    'Omitting InitialCurrents_A must retain the zero DAC seed.');
check_initial_objective(zero,ctx);

% A channel vector such as a static B0 result is repeated across all lines.
static_A = zeros(ctx.nc,1);
static_A(1:2) = [-0.2;0.15];
config.InitialCurrents_A = static_A;
static = run_experiment_fast_parallel_worker(ctx,100,0,1,[],config);
expected_A = repmat(static_A,1,ctx.nstates);
assert(isequal(static.initial_currents_A,expected_A) && ...
    norm(static.currents_A-expected_A,'fro')<1e-12, ...
    'The zero-iteration solver did not receive the repeated static seed.');
check_initial_objective(static,ctx);
assert(abs(static.initial_relative_mse-zero.initial_relative_mse)>1e-6, ...
    'The fixture must distinguish static and zero initialization.');

% Explicit matrices preserve different DAC guesses for different states.
matrix_A = expected_A;
matrix_A(:,2) = matrix_A(:,2)/2;
[prepared_A,~] = prepare_experiment_initial_currents(matrix_A,ctx,30,[100 200]);
assert(isequal(prepared_A,matrix_A));
[row_A,~] = prepare_experiment_initial_currents(static_A.',ctx,30,100);
assert(isequal(row_A,expected_A), 'Row and column channel vectors must agree.');
explicit_zero = config;
explicit_zero.InitialCurrents_A = zeros(ctx.nc,ctx.nstates);
zero_matrix = run_experiment_fast_parallel_worker(ctx,100,0,1,[],explicit_zero);
assert(isequal(zero_matrix.currents_A,zero.currents_A) && ...
    abs(zero_matrix.initial_relative_mse-zero.initial_relative_mse)<1e-14, ...
    'Explicit zero commands differ from the legacy default.');

% The selected seed changes DAC guesses, never the physical initial current.
response_ctx = synthetic_context(true);
response = run_experiment_fast_parallel_worker(response_ctx,100,0,1,[],config);
assert(isequal(response.initial_currents_A,expected_A));
check_initial_objective(response,response_ctx);
assert(response.current_response.enabled && ...
    isequal(response.current_response,response_ctx.current_response) && ...
    abs(response.initial_relative_mse-static.initial_relative_mse)>1e-8, ...
    'Seeding bypassed or changed the first-order physical response.');
[actual_A,command_A] = acdc_sample_current_waveform(expected_A, ...
    response_ctx.line_start_s,response_ctx.line_end_s, ...
    response_ctx.line_start_s(1),response_ctx.current_response);
assert(max(abs(command_A(:)-expected_A(:,1)))<1e-14);
assert(all(actual_A(:)==0), 'The real current before the first command must remain zero.');

% Invalid guesses fail visibly; none is silently clipped or replaced.
expect_failure(@() prepare_experiment_initial_currents(ones(2,2),ctx,30,100), ...
    'prepare_experiment_initial_currents:Shape');
expect_failure(@() prepare_experiment_initial_currents(NaN(ctx.nc,1),ctx,30,100),'');
expect_failure(@() prepare_experiment_initial_currents(Inf(ctx.nc,1),ctx,30,100),'');
expect_failure(@() prepare_experiment_initial_currents(1i*ones(ctx.nc,1),ctx,30,100),'');
over_channel_A = static_A;
over_channel_A(3) = 1.01;
expect_failure(@() prepare_experiment_initial_currents(over_channel_A,ctx,30,100), ...
    'prepare_experiment_initial_currents:ChannelLimit');
expect_failure(@() prepare_experiment_initial_currents(static_A,ctx,0.35,100), ...
    'prepare_experiment_initial_currents:TotalLimit');
field_max_Hz = max(abs(ctx.B_quality*expected_A),[],'all');
expect_failure(@() prepare_experiment_initial_currents(static_A,ctx,30,[100 field_max_Hz/2]), ...
    'prepare_experiment_initial_currents:FieldLimit');
bad_config = config;
bad_config.InitialCurrents_A = over_channel_A;
expect_failure(@() run_experiment_fast_parallel_worker(ctx,100,0,1,[],bad_config), ...
    'prepare_experiment_initial_currents:ChannelLimit');

report = struct('zero_initial_mse',zero.initial_relative_mse, ...
    'static_initial_mse',static.initial_relative_mse, ...
    'first_order_static_initial_mse',response.initial_relative_mse, ...
    'legacy_zero_preserved',true,'zero_iteration_static_seed_verified',true, ...
    'physical_initial_current_zero',true,'invalid_seeds_rejected',true);
if run_bridge, report.bridge = validate_bridge(static_A); end
fprintf(['Image initialization validation: zero MSE %.9g; static seed MSE %.9g; ' ...
    'first-order seed MSE %.9g; actual initial commands and invalid-seed rejection passed.\n'], ...
    report.zero_initial_mse,report.static_initial_mse,report.first_order_static_initial_mse);
end

function report = validate_bridge(static_A)
[ctx,experiment,calibration] = synthetic_context(true);
folder = tempname('/tmp');
mkdir(folder);
cleanup = onCleanup(@() rmdir(folder,'s')); %#ok<NASGU>
input_file = fullfile(folder,'input.mat');
calibration_file = fullfile(folder,'calibration.mat');
save(input_file,'experiment');
save(calibration_file,'-struct','calibration');
request = struct('input_file',input_file,'calibration_file',calibration_file, ...
    'max_current_A',1,'total_current_limit_A',30,'field_limit_Hz',100, ...
    'max_iterations',0,'workers',1,'threads_per_worker',1,'device','cpu', ...
    'scratch_dir',folder);
baseline_image = [];
for use_static = [false true]
    if use_static
        request.initial_currents_A = static_A;
        request.output_dir = fullfile(folder,'static');
        expected_A = repmat(static_A,1,ctx.nstates);
    else
        request.output_dir = fullfile(folder,'zero');
        expected_A = zeros(ctx.nc,ctx.nstates);
    end
    request_file = fullfile(folder,'request.mat');
    save(request_file,'request');
    [result_file,status] = run_acdc_notebook_stage(request_file,'optimize');
    assert(strcmp(status.state,'complete'), 'Bridge failed: %s',status.error_report);
    stage = load(result_file,'summary');
    saved = load(stage.summary.files{1});
    assert(isequal(saved.initial_currents_A,expected_A) && ...
        norm(saved.currents_A-expected_A,'fro')<1e-12 && ...
        isequal(stage.summary.initial_currents_A,expected_A), ...
        'The MAT bridge or wrapper changed the requested solver seed.');
    check_initial_objective(saved,ctx);
    baseline = load(fullfile(status.run_dir,'baseline.mat'),'initial_image');
    if use_static
        assert(isequal(baseline.initial_image,baseline_image), ...
            'Static solver initialization changed the zero-shim baseline.');
        report.static_initial_mse = saved.initial_relative_mse;
    else
        baseline_image = baseline.initial_image;
        report.zero_initial_mse = saved.initial_relative_mse;
    end
end
% Invalid initial commands fail before an optimization folder or pool exists.
request.initial_currents_A = 2*ones(ctx.nc,1);
request.output_dir = fullfile(folder,'invalid');
save(request_file,'request');
[~,status] = run_acdc_notebook_stage(request_file,'optimize');
assert(strcmp(status.state,'failed') && ...
    strcmp(status.error_identifier,'prepare_experiment_initial_currents:ChannelLimit') && ...
    ~isfolder(fullfile(request.output_dir,'optimization')) && isempty(gcp('nocreate')), ...
    'Invalid bridge seed started an optimizer or lost its readable error.');
report.zero_and_static_saved = true;
report.invalid_rejected_before_pool = true;
report.zero_baseline_unchanged = true;
fprintf('Image initialization MAT bridge: zero/static roundtrips and saved seeds passed; baseline unchanged.\n');
end

function check_initial_objective(result,ctx)
expected = obj_fun_experiment(result.initial_currents_A(:),ctx);
assert(abs(result.initial_relative_mse-expected)<1e-12 && ...
    result.history.iteration(1)==0 && ...
    abs(result.history.fval(1)-expected)<1e-12, ...
    'Saved initial MSE and solver iteration zero do not describe the selected seed.');
end

function expect_failure(action,identifier)
failed = false;
try
    action();
catch err
    failed = true;
    assert(isempty(identifier) || strcmp(err.identifier,identifier), ...
        'An invalid seed failed for the wrong reason: %s',err.message);
end
assert(failed, 'An invalid initial-current seed was accepted.');
end

function [ctx,experiment,calibration] = synthetic_context(response_enabled)
object_size = [4 4];
[X,Y] = meshgrid((-2:1)*0.2/4);
m0 = zeros(object_size);
points = [sub2ind(object_size,2,2),sub2ind(object_size,3,4)];
m0(points) = [1 0.7];
b0 = zeros(object_size);
b0(points) = [60 -80];
starts = 0.003+(0:3)'*0.002;
time = starts+0.0001+(0:3)*0.00005;
time(2:2:end,:) = fliplr(time(2:2:end,:));
[kx,ky] = meshgrid((-2:1)/0.2);
experiment = struct('m0_proxy',m0,'geometry',struct('X_m',X,'Y_m',Y), ...
    'field',struct('b0Hz',b0,'mask',m0>0,'simulation_mask',m0>0), ...
    'relaxation',struct('T2star_s',Inf), ...
    'current_response',struct('enabled',response_enabled,'tau_s',linspace(20e-6,50e-6,32)), ...
    'acquisition',struct('kx_mInv',kx,'ky_mInv',ky,'time_s',time, ...
        'TE_s',0.006,'line_start_s',starts,'line_index',repmat((1:4)',1,4)));
maps = zeros([object_size 32]);
for channel = 1:32
    map = zeros(object_size);
    map(points) = [80+channel,(-1)^channel*(40+channel)];
    maps(:,:,channel) = map;
end
calibration = struct('X_m',X,'Y_m',Y,'units','Hz/A','acdc_maps',maps);
ctx = prepare_experiment_optimizer(experiment,calibration,1);
end
