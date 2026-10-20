function result = run_experiment_fast_parallel_worker(ctx_constant, field_limit_Hz, max_iterations, threads, progress_queue, config)
%RUN_EXPERIMENT_FAST_PARALLEL_WORKER Solve one independently constrained limit.
%   No file writes or figures occur on a worker. A plain context is accepted
%   for a short serial validation; normally use a parallel.pool.Constant.
if nargin<5, progress_queue = []; end
if nargin<6, config = struct; end
if ~isfield(config,'Device'), config.Device = 'cpu'; end
if ~isfield(config,'BlockSize'), config.BlockSize = []; end
if ~isfield(config,'TotalCurrentLimit_A'), config.TotalCurrentLimit_A = 30; end
if ~isfield(config,'PruneFieldConstraints'), config.PruneFieldConstraints = true; end
if ~isfield(config,'SubproblemAlgorithm'), config.SubproblemAlgorithm = 'cg'; end
if ~isfield(config,'TargetMSE'), config.TargetMSE = []; end
if ~isfield(config,'InitialCurrents_A'), config.InitialCurrents_A = []; end
if ~isempty(config.TargetMSE)
    validateattributes(config.TargetMSE,{'numeric'},{'real','scalar','finite','positive'});
end
validateattributes(threads,{'numeric'},{'real','scalar','finite','integer','>=',1,'<=',6});
validateattributes(field_limit_Hz,{'numeric'},{'real','scalar','finite','positive'});
validateattributes(max_iterations,{'numeric'},{'real','scalar','finite','integer','nonnegative'});
previous_threads = maxNumCompThreads(threads);
thread_cleanup = onCleanup(@() maxNumCompThreads(previous_threads)); %#ok<NASGU>
if isa(ctx_constant,'parallel.pool.Constant'), ctx = ctx_constant.Value;
else, ctx = ctx_constant; end
[initial_currents_A,initialization_mode] = prepare_experiment_initial_currents( ...
    config.InitialCurrents_A,ctx,config.TotalCurrentLimit_A,field_limit_Hz);
timer = tic;
% Numeric diagnostics let notebooks plot solver progress without parsing text.
history = struct('iteration', [], 'fval', [], 'firstorderopt', [], ...
    'constrviolation', [], 'function_evaluations', [], 'elapsed_seconds', []);
target_stop_requested = false;
if ~isfield(ctx,'acceleration')
    ctx = prepare_experiment_runtime(ctx,'Device',config.Device,'BlockSize',config.BlockSize);
end
objective = @(x) obj_fun_experiment_accelerated(x,ctx);
x0 = initial_currents_A(:);
initial_relative_mse = objective(x0);
% Limit the GENERATED ACDC field, not B0+ACDC, on measured-quality voxels.
% Every state has its own field/current bounds, as in run_code4_fast_parallel.
[A,b,constraint_info] = prepare_experiment_field_constraints( ...
    ctx,field_limit_Hz,config.TotalCurrentLimit_A);
if ~config.PruneFieldConstraints
    % Reference system for timing comparisons; exactly the same feasible set.
    A = kron(speye(ctx.nstates),sparse(ctx.B_quality)); A = [A;-A];
    b = field_limit_Hz*ones(size(A,1),1);
    constraint_info.nKeptVoxels = size(ctx.B_quality,1);
    constraint_info.rowsAfter = size(A,1);
    constraint_info.method = 'Full field constraints (reference benchmark)';
end
upper = repmat(ctx.Imax_A,ctx.nstates,1);
current_constraint = @(x) experiment_current_constraint( ...
    x,ctx.nc,ctx.nstates,config.TotalCurrentLimit_A);
options = optimoptions('fmincon','Algorithm','interior-point', ...
    'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,'Display','iter', ...
    'SubproblemAlgorithm',config.SubproblemAlgorithm, ...
    'MaxIterations',max_iterations,'MaxFunctionEvaluations',1e7, ...
    'OptimalityTolerance',1e-6,'StepTolerance',1e-6,'ConstraintTolerance',1e-8, ...
    'OutputFcn',@progress);
% Predeclare evalc outputs for this function's nested callback workspace.
x = []; value = []; flag = []; output = [];
solver_log = evalc('[x,value,flag,output] = fmincon(objective,x0,A,b,[],[],-upper,upper,current_constraint,options);');
currents_A = reshape(x,ctx.nc,ctx.nstates);
[image,kdata] = forward_model_experiment_accelerated(currents_A,ctx);
if isa(image,'gpuArray'), image = gather(image); kdata = gather(kdata); end
absolute_current_sum_A = sum(abs(currents_A),1);
current_limits_met = all(all(abs(currents_A)<=ctx.Imax_A+1e-8)) && ...
    all(absolute_current_sum_A<config.TotalCurrentLimit_A);
maximum_quality_field_Hz = max(abs(ctx.B_quality*currents_A),[],'all');
field_limits_met = maximum_quality_field_Hz<=field_limit_Hz+1e-8;
% These additional checks evaluate predicted real currents at ADC samples.
% The optimizer's existing constraints still apply to the DAC commands.
response_diagnostics = acdc_response_diagnostics( ...
    currents_A,ctx,field_limit_Hz,config.TotalCurrentLimit_A);
actual_adc_limits_met = response_diagnostics.actual_adc_field_limits_met && ...
    response_diagnostics.actual_adc_channel_limits_met && ...
    response_diagnostics.actual_adc_current_sum_limits_met;
target_reached = experiment_mse_target_reached(x,value,ctx,field_limit_Hz,config);
target_stop = flag==-1 && target_stop_requested && target_reached;
accepted = (flag>0 || target_stop) && current_limits_met && field_limits_met;
if target_stop
    stopping_reason = 'Target MSE reached with current and field limits satisfied.';
elseif flag>0 && accepted
    stopping_reason = 'Solver convergence with current and field limits satisfied.';
else
    stopping_reason = 'Solver ended without an accepted solution.';
end
result = struct('currents_A',currents_A,'current_response',ctx.current_response, ...
    'initial_currents_A',initial_currents_A,'initial_relative_mse',initial_relative_mse, ...
    'initialization_mode',initialization_mode, ...
    'response_diagnostics',response_diagnostics,'generated_field_Hz', ...
    reshape(ctx.B_full*currents_A,[ctx.object_size ctx.nstates]), ...
    'image',image,'kdata',kdata,'field_limit_Hz',field_limit_Hz, ...
    'optimization_info',struct('fval',value,'exitflag',flag,'output',output, ...
        'current_limits_met',current_limits_met,'field_limits_met',field_limits_met, ...
        'converged',flag>0 && current_limits_met && field_limits_met, ...
        'target_mse',config.TargetMSE,'target_reached',target_reached, ...
        'accepted',accepted,'stopping_reason',stopping_reason), ...
    'absolute_current_sum_A',absolute_current_sum_A, ...
    'TotalCurrentLimit_A',config.TotalCurrentLimit_A, ...
    'maximum_quality_field_Hz',maximum_quality_field_Hz, ...
    'constraint_info',constraint_info,'acceleration',ctx.acceleration, ...
    'subproblem_algorithm',config.SubproblemAlgorithm, ...
    'TargetMSE',config.TargetMSE, ...
    'elapsed_seconds',toc(timer),'solver_log',solver_log,'history',history, ...
    'worker_threads',maxNumCompThreads);
result.optimization_info.actual_adc_field_limits_met = response_diagnostics.actual_adc_field_limits_met;
result.optimization_info.actual_adc_current_limits_met = ...
    response_diagnostics.actual_adc_channel_limits_met && ...
    response_diagnostics.actual_adc_current_sum_limits_met;
result.optimization_info.actual_adc_limits_met = actual_adc_limits_met;
result.optimization_info.constraint_domain = 'DAC commands; predicted real currents checked separately at ADC samples';

    function stop = progress(candidate,values,state)
        stop = false;
        if strcmp(state,'init') || strcmp(state,'iter')
            % OutputFcn runs after the initial objective evaluation and every
            % iteration. Send these numeric diagnostics to the MATLAB client;
            % evalc still captures the worker's native fmincon solver log.
            data = struct('limit_Hz',field_limit_Hz,'state',state, ...
                'elapsed_seconds',toc(timer));
            fields = {'iteration','fval','firstorderopt','constrviolation','funccount'};
            for jj = 1:numel(fields)
                data.(fields{jj}) = NaN;
                if isfield(values,fields{jj}), data.(fields{jj}) = values.(fields{jj}); end
            end
            data.function_evaluations = data.funccount;
            data = rmfield(data,'funccount');
            history_fields = fieldnames(history);
            for jj = 1:numel(history_fields)
                key = history_fields{jj};
                history.(key)(end+1) = data.(key);
            end
            if ~isempty(progress_queue), send(progress_queue,data); end
            % An adequate image can end the solve early, but a low objective
            % alone must never accept an infeasible current waveform.
            stop = experiment_mse_target_reached( ...
                candidate,data.fval,ctx,field_limit_Hz,config);
            if stop
                target_stop_requested = true;
                fprintf('Target MSE reached: %.8g < %.8g; DAC current and field limits satisfied.\n', ...
                    data.fval,config.TargetMSE);
            end
        end
    end
end
