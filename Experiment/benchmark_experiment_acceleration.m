function report = benchmark_experiment_acceleration(ctx, varargin)
%BENCHMARK_EXPERIMENT_ACCELERATION Compare exact objective/gradient backends.
%   REPORT = benchmark_experiment_acceleration(CTX,'UseGPU',true) requires a
%   CPU context from prepare_experiment_optimizer. This does NOT optimize or
%   program currents. All computations retain double precision. The cache
%   transfer to GPU is timed separately and excluded from steady-state time.
%   Options: Threads [1 2 4 6], BlockSize [] (backend default), UseGPU false, Currents []
%   (zero starting currents). No pool is created and no GPU reset is made.
parser = inputParser;
parser.PartialMatching = false;
addParameter(parser,'Threads',[1 2 4 6],@(v) isnumeric(v) && isreal(v) && ...
    isvector(v) && ~isempty(v) && all(isfinite(v)) && all(v==fix(v)) && ...
    all(v>=1 & v<32));
addParameter(parser,'BlockSize',[],@(v) isempty(v) || (isnumeric(v) && isreal(v) && ...
    isscalar(v) && isfinite(v) && v==fix(v) && v>=1));
addParameter(parser,'UseGPU',false,@(v) islogical(v) && isscalar(v));
addParameter(parser,'Currents',[],@(v) isnumeric(v) && isreal(v) && all(isfinite(v(:))));
parse(parser,varargin{:});
options = parser.Results;
assert(~isa(ctx.B,'gpuArray') && ~isa(ctx.base_signal,'gpuArray'), ...
    'Supply the original CPU context; a resident GPU cache is prepared here.');
x = options.Currents(:);
if isempty(x), x = zeros(ctx.nc*ctx.nstates,1); end
assert(numel(x)==ctx.nc*ctx.nstates,'Currents must have one value per channel/state.');
previous_threads = maxNumCompThreads;
thread_cleanup = onCleanup(@() maxNumCompThreads(previous_threads)); %#ok<NASGU>
report = struct('matlab_release',version('-release'),'precision','double', ...
    'nvoxels',size(ctx.B,1),'nvariables',numel(x),'nsamples',size(ctx.H_s,1), ...
    'block_size',options.BlockSize,'cpu',[],'gpu',[], ...
    'timing_scope','Objective plus analytic gradient; no constraints or solver');
for threads = options.Threads(:).'
    maxNumCompThreads(threads);
    runtime = prepare_experiment_runtime(ctx,'BlockSize',options.BlockSize,'Device','cpu');
    [original_value,original_gradient] = obj_fun_experiment(x,ctx);
    [value,gradient] = obj_fun_experiment_accelerated(x,runtime);
    value_error = abs(value-original_value)/max(1,abs(original_value));
    gradient_error = norm(gradient-original_gradient)/max(1,norm(original_gradient));
    assert(value_error<1e-11 && gradient_error<1e-10, ...
        'Accelerated CPU objective/gradient differs from the original.');
    original_s = timeit(@() obj_fun_experiment(x,ctx),2);
    accelerated_s = timeit(@() obj_fun_experiment_accelerated(x,runtime),2);
    row = struct('threads',maxNumCompThreads,'block_size',runtime.acceleration.BlockSize, ...
        'original_seconds',original_s, ...
        'accelerated_seconds',accelerated_s,'speedup',original_s/accelerated_s, ...
        'relative_value_error',value_error,'relative_gradient_error',gradient_error);
    report.cpu = [report.cpu row]; %#ok<AGROW>
    fprintf('CPU %d threads: original %.6g s; accelerated %.6g s; %.3gx.\n', ...
        row.threads,original_s,accelerated_s,row.speedup);
end
if options.UseGPU
    % GPU selection without an index preserves the session's selected device.
    % Neither reset(device) nor gpuDevice(index) is used by this benchmark.
    assert(license('test','Distrib_Computing_Toolbox'), ...
        'GPU computation requires Parallel Computing Toolbox.');
    device = gpuDevice;
    maxNumCompThreads(1);
    wait(device);
    started = tic;
    runtime_gpu = prepare_experiment_runtime(ctx,'BlockSize',options.BlockSize,'Device','gpu');
    wait(device);
    transfer_seconds = toc(started);
    [value,gradient] = obj_fun_experiment_accelerated(x,runtime_gpu);
    value_error = abs(value-original_value)/max(1,abs(original_value));
    gradient_error = norm(gradient-original_gradient)/max(1,norm(original_gradient));
    assert(isa(value,'double') && isa(gradient,'double') && ...
        ~isa(value,'gpuArray') && ~isa(gradient,'gpuArray'), ...
        'The solver interface must return ordinary CPU double outputs.');
    assert(value_error<1e-11 && gradient_error<1e-10, ...
        'GPU double objective/gradient differs from the original.');
    gpu_seconds = gputimeit(@() obj_fun_experiment_accelerated(x,runtime_gpu),2);
    [cpu_seconds,best] = min([report.cpu.accelerated_seconds]);
    report.gpu = struct('name',device.Name,'compute_capability',device.ComputeCapability, ...
        'block_size',runtime_gpu.acceleration.BlockSize, ...
        'toolkit_version',device.ToolkitVersion,'total_memory_bytes',device.TotalMemory, ...
        'available_memory_bytes',device.AvailableMemory,'single_double_ratio',device.SingleDoubleRatio, ...
        'transfer_seconds',transfer_seconds,'objective_gradient_seconds',gpu_seconds, ...
        'speedup_vs_best_cpu',cpu_seconds/gpu_seconds,'best_cpu_threads',report.cpu(best).threads, ...
        'relative_value_error',value_error,'relative_gradient_error',gradient_error);
    fprintf('GPU double: %.6g s; %.3gx vs best CPU (%d threads); transfer %.3g s.\n', ...
        gpu_seconds,cpu_seconds/gpu_seconds,report.cpu(best).threads,transfer_seconds);
end
end
