function report = validate_experiment_acceleration(ctx, varargin)
%VALIDATE_EXPERIMENT_ACCELERATION Compare exact CPU/GPU paths with old model.
%   With no CTX, uses an in-memory synthetic fixture. No calibration or
%   simulated current is saved or presented as an experimental measurement.
%   An explicit measured CTX may be supplied for numerical checks only.
%   Options: Devices ({'cpu'}), BlockSizes ([32 128 256 1024]).
parser = inputParser;
parser.PartialMatching = false;
addParameter(parser,'Devices',{'cpu'},@(v) iscellstr(v) && ~isempty(v));
addParameter(parser,'BlockSizes',[32 128 256 1024],@(v) isnumeric(v) && ...
    isvector(v) && all(isfinite(v)) && all(v>=1) && all(v==fix(v)));
parse(parser,varargin{:});
previous_rng = rng;
rng_cleanup = onCleanup(@() rng(previous_rng)); %#ok<NASGU>
rng(483);
synthetic = nargin<1 || isempty(ctx);
if synthetic, ctx = synthetic_fixture(); end
nx = ctx.nc*ctx.nstates;
currents = {zeros(nx,1),.001*randn(nx,1),.003*randn(nx,1)};
template = struct('device','','block_size',0,'value_relative_error',0, ...
    'gradient_relative_error',0,'image_relative_error',0,'kdata_relative_error',0);
cases = repmat(template,0,1);
for device = parser.Results.Devices
    for block_size = parser.Results.BlockSizes
        runtime = prepare_experiment_runtime(ctx,'Device',device{1},'BlockSize',block_size);
        assert(runtime.acceleration.TimingStructured, 'Fixture/experiment must have exact imaging-line timing.');
        row = template;
        row.device = device{1}; row.block_size = block_size;
        for test = 1:numel(currents)
            x = currents{test};
            [old_value,old_gradient] = obj_fun_experiment(x,ctx);
            [new_value,new_gradient] = obj_fun_experiment_accelerated(x,runtime);
            new_value_only = obj_fun_experiment_accelerated(x,runtime);
            assert(abs(new_value_only-new_value)<=1e-13*max(1,abs(new_value)), ...
                'Objective-only and fused objective disagree.');
            [old_image,old_kdata] = forward_model_experiment_acdc(reshape(x,ctx.nc,ctx.nstates),ctx);
            [new_image,new_kdata] = forward_model_experiment_accelerated(reshape(x,ctx.nc,ctx.nstates),runtime);
            new_image = cpu_value(new_image); new_kdata = cpu_value(new_kdata);
            row.value_relative_error = max(row.value_relative_error, ...
                abs(new_value-old_value)/max(abs(old_value),1e-12));
            row.gradient_relative_error = max(row.gradient_relative_error, ...
                norm(new_gradient-old_gradient)/max(norm(old_gradient),1e-12));
            row.image_relative_error = max(row.image_relative_error, ...
                norm(new_image(:)-old_image(:))/max(norm(old_image(:)),1e-12));
            row.kdata_relative_error = max(row.kdata_relative_error, ...
                norm(new_kdata(:)-old_kdata(:))/max(norm(old_kdata(:)),1e-12));
            assert(isfinite(new_value) && all(isfinite(new_gradient)), 'Nonfinite fused result.');
        end
        assert(max([row.value_relative_error row.gradient_relative_error ...
            row.image_relative_error row.kdata_relative_error])<1e-10, ...
            'Accelerated model differs materially from the original double model.');
        cases(end+1,1) = row; %#ok<AGROW>
        fprintf('Acceleration %s block %d: objective %.3g, gradient %.3g, image %.3g.\n', ...
            row.device,row.block_size,row.value_relative_error, ...
            row.gradient_relative_error,row.image_relative_error);
    end
end

% Directional checks include an early state and the final imaging state,
% checking phase history, adjoint FFT scaling and magnitude differentiation.
runtime = prepare_experiment_runtime(ctx,'Device','cpu','BlockSize',128);
x = currents{2};
[~,gradient] = obj_fun_experiment_accelerated(x,runtime);
directions = randn(nx,2);
directions(:,3:4) = 0;
directions(1,3) = 1;
directions((ctx.nstates-1)*ctx.nc+ctx.nc,4) = 1;
errors = zeros(1,size(directions,2));
for test = 1:size(directions,2)
    direction = directions(:,test)/norm(directions(:,test));
    step = 1e-6;
    numerical = (obj_fun_experiment_accelerated(x+step*direction,runtime) ...
        -obj_fun_experiment_accelerated(x-step*direction,runtime))/(2*step);
    analytical = gradient.'*direction;
    errors(test) = abs(numerical-analytical)/max([abs(numerical),abs(analytical),1e-8]);
end
assert(all(errors<2e-6), 'Fused adjoint failed independent directional checks.');

% An unusual H matrix must use the exact generic fallback, never approximate
% it by the imaging-line staircase. This is a mathematical fixture only.
unusual = ctx;
unusual.H_s(1,end) = unusual.H_s(1,end)+1e-5;
runtime = prepare_experiment_runtime(unusual,'Device','cpu','BlockSize',128);
assert(~runtime.acceleration.TimingStructured, 'Non-staircase timing was incorrectly simplified.');
[old_value,old_gradient] = obj_fun_experiment(x,unusual);
[new_value,new_gradient] = obj_fun_experiment_accelerated(x,runtime);
fallback_error = max(abs(old_value-new_value)/max(abs(old_value),1e-12), ...
    norm(old_gradient-new_gradient)/max(norm(old_gradient),1e-12));
assert(fallback_error<1e-10, 'Generic fused timing fallback differs from the original.');
report = struct('synthetic_fixture',synthetic,'cases',cases, ...
    'directional_gradient_relative_errors',errors, ...
    'generic_fallback_relative_error',fallback_error, ...
    'original_context_unchanged',~isempty(ctx.base_signal),'status','passed');
fprintf('Acceleration independent gradients max %.3g; generic fallback %.3g; synthetic %d.\n', ...
    max(errors),fallback_error,synthetic);
end

function value = cpu_value(value)
if isa(value,'gpuArray'), value = gather(value); end
end

function ctx = synthetic_fixture()
% Deliberately sparse in-memory support, retaining all actual matrix sizes.
object_size = [128 128];
[X,Y] = meshgrid(((0:127)-64)*.2/128);
experiment = struct;
experiment.geometry = struct('X_m',X,'Y_m',Y);
experiment.m0_proxy = zeros(object_size);
points = randperm(prod(object_size),137);
experiment.m0_proxy(points) = .2+rand(size(points));
experiment.field = struct('b0Hz',zeros(object_size), ...
    'mask',experiment.m0_proxy>0,'simulation_mask',experiment.m0_proxy>0);
experiment.field.b0Hz(points) = 45*randn(size(points));
experiment.relaxation.T2star_s = .032;
starts = .005+cumsum([0; .0004+.0002*rand(31,1)]);
width = min(diff(starts))/34;
time = starts+width*(1:32);
time(2:2:end,:) = fliplr(time(2:2:end,:));
[kx,ky] = meshgrid((-16:15)/.2);
experiment.acquisition = struct('kx_mInv',kx,'ky_mInv',ky,'time_s',time, ...
    'TE_s',.012,'line_start_s',starts,'line_index',repmat((1:32)',1,32));
calibration = struct('X_m',X,'Y_m',Y,'units','Hz/A', ...
    'acdc_maps',reshape(110*randn(prod(object_size),32),[object_size 32]));
ctx = prepare_experiment_optimizer(experiment,calibration,.2);
end
