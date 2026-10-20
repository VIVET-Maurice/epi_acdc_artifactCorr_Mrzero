function report = validate_experiment_mse_stop
%VALIDATE_EXPERIMENT_MSE_STOP Check early stopping with a small real solver.
%   All images, fields and currents here are synthetic validation fixtures.
%   This function does not write files or start a parallel pool.
ctx = synthetic_context([60 -80]);
config = struct('Device','cpu','TotalCurrentLimit_A',30,'TargetMSE',0.005);
initial_mse = obj_fun_experiment(zeros(ctx.nc*ctx.nstates,1),ctx);
assert(initial_mse>config.TargetMSE, 'Fixture must start above the requested MSE.');

% A useful objective is enough to stop, without pretending that fmincon
% independently converged. The callback must preserve its real -1 exit flag.
solution = run_experiment_fast_parallel_worker(ctx,100,80,1,[],config);
info = solution.optimization_info;
assert(info.fval<config.TargetMSE && info.target_reached && info.accepted && ...
    info.exitflag==-1 && ~info.converged, 'Target stop was not accepted correctly.');
assert(solution.history.iteration(end)>0 && ...
    solution.history.iteration(end)<80 && ...
    contains(solution.solver_log,'Target MSE reached'), ...
    'Target stop did not occur before the iteration cap or was not logged.');
report.initial_mse = initial_mse;
report.final_mse = info.fval;
report.target_stop_iteration = solution.history.iteration(end);

% Even the initial point may already satisfy the requested image accuracy.
ideal_ctx = synthetic_context([0 0]);
initial_solution = run_experiment_fast_parallel_worker(ideal_ctx,100,80,1,[],config);
initial_info = initial_solution.optimization_info;
assert(initial_info.exitflag==-1 && initial_info.accepted && ...
    initial_info.target_reached && initial_solution.history.iteration(end)==0, ...
    'An adequate initial point should stop before the first optimization step.');
report.initial_target_stop = true;

% Reaching a solver limit above the target cannot produce an accepted result.
unreachable_config = config;
unreachable_config.TargetMSE = initial_mse/1000;
capped = run_experiment_fast_parallel_worker(ctx,100,0,1,[],unreachable_config);
assert(capped.optimization_info.exitflag==0 && ...
    ~capped.optimization_info.target_reached && ~capped.optimization_info.accepted, ...
    'An unfinished solve above the target must remain unaccepted.');
report.capped_solve_rejected = true;

% Old callers omit TargetMSE and retain ordinary solver convergence.
legacy_config = rmfield(config,'TargetMSE');
legacy = run_experiment_fast_parallel_worker(ideal_ctx,100,80,1,[],legacy_config);
legacy_info = legacy.optimization_info;
assert(isempty(legacy_info.target_mse) && ~legacy_info.target_reached && ...
    legacy_info.exitflag>0 && legacy_info.converged && legacy_info.accepted, ...
    'Disabling the target changed ordinary convergence semantics.');
report.legacy_convergence_preserved = true;

% Trial points may have low MSE while violating physical limits. Check the
% same predicate used by the OutputFcn, including the strict sum boundary.
check_ctx = struct('nc',2,'nstates',1,'Imax_A',[1;1],'B_quality',[100 100]);
check_config = struct('TargetMSE',0.005,'TotalCurrentLimit_A',1);
assert(experiment_mse_target_reached([0;0],0.001,check_ctx,10,check_config));
assert(~experiment_mse_target_reached([1.1;0],0.001,check_ctx,1000,check_config), ...
    'Low MSE bypassed a channel current limit.');
assert(~experiment_mse_target_reached([0.5;0.5],0.001,check_ctx,1000,check_config), ...
    'Low MSE bypassed the strict total-current limit.');
assert(~experiment_mse_target_reached([0.11;0],0.001,check_ctx,10,check_config), ...
    'Low MSE bypassed the generated-field limit.');
assert(~experiment_mse_target_reached([0;0],0.005,check_ctx,10,check_config) && ...
    ~experiment_mse_target_reached([0;0],NaN,check_ctx,10,check_config), ...
    'The target must use a finite MSE strictly below the threshold.');
report.infeasible_low_mse_rejected = true;
fprintf('MSE stopping validation: %.6g -> %.6g in %d iterations; initial stop, legacy convergence, cap and physical bounds passed.\n', ...
    report.initial_mse,report.final_mse,report.target_stop_iteration);
end

function ctx = synthetic_context(b0_Hz)
object_size = [4 4];
[X,Y] = meshgrid((-2:1)*0.2/4);
m0 = zeros(object_size);
points = [sub2ind(object_size,2,2),sub2ind(object_size,3,4)];
m0(points) = [1 0.7];
b0 = zeros(object_size);
b0(points) = b0_Hz;
starts = 0.003+(0:3)'*0.002;
time = starts+0.0001+(0:3)*0.00005;
time(2:2:end,:) = fliplr(time(2:2:end,:));
[kx,ky] = meshgrid((-2:1)/0.2);
experiment = struct('m0_proxy',m0,'geometry',struct('X_m',X,'Y_m',Y), ...
    'field',struct('b0Hz',b0,'mask',m0>0,'simulation_mask',m0>0), ...
    'relaxation',struct('T2star_s',Inf), ...
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
