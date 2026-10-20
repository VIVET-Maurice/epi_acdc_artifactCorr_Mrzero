function report = benchmark_experiment_solver_options(ctx, varargin)
%BENCHMARK_EXPERIMENT_SOLVER_OPTIONS Short identical-problem solver comparison.
%   No pool, physical current file or full optimization is created. Reports
%   timings and objective progress; a short test cannot establish convergence.
%   Options: Device ('cpu'), Threads (6), MaxIterations (5), FieldLimit (100),
%   TotalCurrentLimit_A (30), BlockSize ([]), Cases (all four variants).
%   https://www.mathworks.com/help/optim/ug/fmincon.html documents BFGS,
%   limited-memory BFGS and factorization/CG interior-point subproblems.
parser = inputParser;
parser.PartialMatching = false;
addParameter(parser,'Device','cpu',@(v) ischar(v) || (isstring(v) && isscalar(v)));
addParameter(parser,'Threads',6,@(v) isnumeric(v) && isscalar(v) && isfinite(v) && v>=1 && v<=6 && v==fix(v));
addParameter(parser,'MaxIterations',5,@(v) isnumeric(v) && isscalar(v) && isfinite(v) && v>=1 && v==fix(v));
addParameter(parser,'FieldLimit',100,@(v) isnumeric(v) && isscalar(v) && isfinite(v) && v>0);
addParameter(parser,'TotalCurrentLimit_A',30,@(v) isnumeric(v) && isscalar(v) && isfinite(v) && v>1e-4);
addParameter(parser,'BlockSize',[],@(v) isempty(v) || (isnumeric(v) && isscalar(v) && v>=1 && v==fix(v)));
addParameter(parser,'Cases',{'bfgs-factorization','bfgs-cg','lbfgs-auto','lbfgs-cg'},@iscellstr);
parse(parser,varargin{:});
config = parser.Results;
previous_threads = maxNumCompThreads(config.Threads);
cleanup = onCleanup(@() maxNumCompThreads(previous_threads)); %#ok<NASGU>
runtime = prepare_experiment_runtime(ctx,'Device',config.Device,'BlockSize',config.BlockSize);
[A,b,field_info] = prepare_experiment_field_constraints(ctx,config.FieldLimit,config.TotalCurrentLimit_A);
upper = repmat(ctx.Imax_A,ctx.nstates,1);
x0 = zeros(ctx.nc*ctx.nstates,1);
objective = @(x) obj_fun_experiment_accelerated(x,runtime);
constraint = @(x) experiment_current_constraint(x,ctx.nc,ctx.nstates,config.TotalCurrentLimit_A);
[initial_value,~] = objective(x0); % warm the execution path before timing
options_base = optimoptions('fmincon','Algorithm','interior-point', ...
    'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,'Display','iter', ...
    'MaxIterations',config.MaxIterations,'MaxFunctionEvaluations',1e7, ...
    'OptimalityTolerance',1e-6,'StepTolerance',1e-6,'ConstraintTolerance',1e-8, ...
    'OutputFcn',@progress);
template = struct('name','','seconds',0,'fval',NaN,'iterations',0,'func_count',0, ...
    'exitflag',0,'constraint_violation',NaN,'first_order_optimality',NaN, ...
    'max_quality_field_Hz',NaN,'max_absolute_current_sum_A',NaN, ...
    'history',[],'solver_log','','output',[]);
rows = repmat(template,numel(config.Cases),1);
history = [];
x = []; value = []; flag = []; output = []; % nested workspace/evalc outputs
for ii = 1:numel(config.Cases)
    name = validatestring(config.Cases{ii},{'bfgs-factorization','bfgs-cg','lbfgs-auto','lbfgs-cg'});
    options = options_base;
    switch name
        case 'bfgs-factorization'
            options.HessianApproximation = 'bfgs'; options.SubproblemAlgorithm = 'factorization';
        case 'bfgs-cg'
            options.HessianApproximation = 'bfgs'; options.SubproblemAlgorithm = 'cg';
        case 'lbfgs-auto'
            options.HessianApproximation = 'lbfgs';
        case 'lbfgs-cg'
            options.HessianApproximation = 'lbfgs'; options.SubproblemAlgorithm = 'cg';
    end
    history = zeros(0,3);
    timer = tic;
    solver_log = evalc('[x,value,flag,output] = fmincon(objective,x0,A,b,[],[],-upper,upper,constraint,options);');
    seconds = toc(timer);
    currents = reshape(x,ctx.nc,ctx.nstates);
    max_field = max(abs(ctx.B_quality*currents),[],'all');
    max_current = max(sum(abs(currents),1));
    assert(max_field<=config.FieldLimit+1e-8 && max_current<config.TotalCurrentLimit_A && ...
        all(abs(currents)<=ctx.Imax_A+1e-8,'all'), 'Short solver variant violates the original limits.');
    row = template;
    row.name = name; row.seconds = seconds; row.fval = value;
    row.iterations = output.iterations; row.func_count = output.funcCount; row.exitflag = flag;
    row.constraint_violation = output.constrviolation; row.first_order_optimality = output.firstorderopt;
    row.max_quality_field_Hz = max_field; row.max_absolute_current_sum_A = max_current;
    row.history = history; row.solver_log = solver_log; row.output = output;
    rows(ii) = row;
    fprintf('Solver %-20s %.6gs, %dit, %deval, f %.12g (initial %.12g), maxfield %.6gHz, maxL1 %.6gA.\n', ...
        name,seconds,output.iterations,output.funcCount,value,initial_value,max_field,max_current);
end
report = struct('cases',rows,'config',config,'initial_fval',initial_value, ...
    'nvoxels',size(ctx.B,1),'field_constraint_info',field_info, ...
    'timing_only',true,'convergence_not_established',true, ...
    'registration_confirmed',ctx.registration_confirmed);

    function stop = progress(~,values,state)
        stop = false;
        if strcmp(state,'init') || strcmp(state,'iter')
            history(end+1,:) = [values.iteration values.fval values.constrviolation];
        end
    end
end
