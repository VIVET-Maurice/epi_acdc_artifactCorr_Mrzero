function report = validate_experiment_gre()
%VALIDATE_EXPERIMENT_GRE Independent spoiled-GRE physics and gradient checks.
% Synthetic fixtures only: rectangular/odd grids, phase-encoding permutation,
% variable B0/T2*, nonzero RF-minus-ADC phase and 32 calibrated channels.
previous_rng = rng;
cleanup = onCleanup(@() rng(previous_rng)); %#ok<NASGU>
rng(701);
shapes = {[9 11],[12 8]};
grids = {[4 6],[5 7]};
cases = cell(1,numel(shapes));
for test = 1:numel(shapes)
    [experiment,calibration] = fixture(shapes{test},grids{test});
    ctx = prepare_experiment_optimizer(experiment,calibration,.2);
    assert(isequal(ctx.object_size,shapes{test}) && isequal(ctx.image_size,grids{test}) ...
        && ctx.nstates==grids{test}(1), 'Prepared cache retained fixed EPI dimensions.');
    runtime = prepare_experiment_runtime(ctx,'Device','cpu','BlockSize',7);
    assert(~runtime.acceleration.TimingStructured, 'Independent GRE lines need generic H integration.');
    zero = zeros(ctx.nc,ctx.nstates);
    [zero_image,zero_k] = forward_model_experiment_acdc(zero,ctx);
    [static_image,static_k] = forward_model_experiment(experiment.m0_proxy, ...
        experiment.field.b0Hz,experiment.geometry,experiment.acquisition,experiment.relaxation);
    zero_error = max(relative_error(zero_image,static_image),relative_error(zero_k,static_k));
    assert(zero_error<1e-13, 'Zero GRE currents disagree with the static forward model.');
    ideal = forward_model_experiment(experiment.m0_proxy,zeros(ctx.object_size), ...
        experiment.geometry,experiment.acquisition,experiment.relaxation);
    assert(relative_error(ideal,ctx.reference)<1e-13, 'GRE ideal reference differs from zero-B0 physics.');

    currents = .002*randn(ctx.nc,ctx.nstates);
    [image,kdata] = forward_model_experiment_acdc(currents,ctx);
    [fast_image,fast_k] = forward_model_experiment_accelerated(currents,runtime);
    expected_k = direct_signal(experiment,calibration,currents);
    direct_error = relative_error(kdata,expected_k);
    fast_error = max(relative_error(fast_image,image),relative_error(fast_k,kdata));
    assert(direct_error<1e-12 && fast_error<1e-12, 'GRE signal disagrees with independent per-excitation physics.');
    index = experiment.acquisition.line_index(:);
    assert(all(sum(ctx.H_s>0,2)==1), 'GRE shim phase leaked from a previous RF history.');
    elapsed = experiment.acquisition.time_s(:)-ctx.line_start_s(index);
    assert(max(abs(sum(ctx.H_s,2)-elapsed))<1e-14, 'GRE integration duration is wrong.');

    changed = currents;
    changed(:,2) = changed(:,2)+.04*randn(ctx.nc,1);
    [~,changed_k] = forward_model_experiment_acdc(changed,ctx);
    untouched = experiment.acquisition.line_index~=2;
    independent_error = norm(changed_k(untouched)-kdata(untouched));
    assert(independent_error<1e-12 && norm(changed_k(~untouched)-kdata(~untouched))>1e-5, ...
        'Changing one GRE state affected another excitation or had no effect.');

    % Changing TR gaps cannot change RF-relative decay or field phase in a
    % fully spoiled model. Acquisition order differs from Cartesian row order.
    shifted = experiment;
    gaps = .013*(0:ctx.nstates-1)'.^2;
    per_sample = reshape(gaps(index),ctx.image_size);
    shifted.acquisition.time_s = shifted.acquisition.time_s+per_sample;
    shifted.acquisition.rf_time_s = shifted.acquisition.rf_time_s+per_sample;
    shifted.acquisition.line_start_s = shifted.acquisition.line_start_s+gaps;
    shifted.acquisition.line_end_s = shifted.acquisition.line_end_s+gaps;
    shifted_ctx = prepare_experiment_optimizer(shifted,calibration,.2);
    [~,shifted_k] = forward_model_experiment_acdc(currents,shifted_ctx);
    gap_error = relative_error(shifted_k,kdata);
    assert(gap_error<1e-12, 'GRE incorrectly accumulated phase or decay across TR gaps.');

    without_phase = experiment.acquisition;
    without_phase.phase_offset_rad = zeros(ctx.image_size);
    [~,unphased_k] = forward_model_experiment(experiment.m0_proxy,experiment.field.b0Hz, ...
        experiment.geometry,without_phase,experiment.relaxation);
    phase_error = relative_error(static_k,unphased_k.*exp(1i*experiment.acquisition.phase_offset_rad));
    assert(phase_error<1e-13, 'RF-minus-ADC phase has the wrong sign or sample order.');

    x = currents(:);
    [value,gradient] = obj_fun_experiment_accelerated(x,runtime);
    [old_value,old_gradient] = obj_fun_experiment(x,ctx);
    assert(abs(value-old_value)<1e-12 && relative_error(gradient,old_gradient)<1e-11, ...
        'GRE accelerated and unaccelerated adjoints differ.');
    directions = randn(numel(x),4);
    directions(:,5:6) = 0;
    directions(1,5) = 1;
    directions(end,6) = 1;
    errors = zeros(1,size(directions,2));
    for direction_index = 1:size(directions,2)
        direction = directions(:,direction_index)/norm(directions(:,direction_index));
        step = 1e-6;
        numerical = (obj_fun_experiment_accelerated(x+step*direction,runtime) ...
            -obj_fun_experiment_accelerated(x-step*direction,runtime))/(2*step);
        analytical = gradient.'*direction;
        errors(direction_index) = abs(numerical-analytical)/max([abs(numerical),abs(analytical),1e-8]);
    end
    assert(all(errors<2e-6), 'GRE adjoint failed finite-difference directional checks.');

    bad = experiment;
    bad.acquisition.line_end_s(1) = bad.acquisition.line_start_s(2)+.001;
    expect_failure(bad,calibration);
    bad = experiment;
    bad.acquisition = rmfield(bad.acquisition,'rf_time_s');
    expect_failure(bad,calibration);
    cases{test} = struct('object_size',ctx.object_size,'image_size',ctx.image_size, ...
        'zero_current_relative_error',zero_error,'direct_signal_relative_error',direct_error, ...
        'accelerated_relative_error',fast_error,'other_lines_error',independent_error, ...
        'tr_gap_relative_error',gap_error,'rf_phase_relative_error',phase_error, ...
        'directional_gradient_relative_errors',errors);
    fprintf('GRE object %dx%d, scan %dx%d: static %.3g, direct %.3g, gap %.3g, gradient %.3g.\n', ...
        ctx.object_size,ctx.image_size,zero_error,direct_error,gap_error,max(errors));
end
report = struct('synthetic_fixture',true,'cases',{cases},'status','passed');
end

function [experiment,calibration] = fixture(object_size,grid_size)
[X,Y] = meshgrid(((0:object_size(2)-1)-floor(object_size(2)/2))*.2/object_size(2), ...
    ((0:object_size(1)-1)-floor(object_size(1)/2))*.18/object_size(1));
experiment.geometry = struct('X_m',X,'Y_m',Y);
experiment.m0_proxy = .2+rand(object_size);
experiment.m0_proxy(1,1) = 0;
experiment.field = struct('b0Hz',35*randn(object_size), ...
    'mask',experiment.m0_proxy>0,'simulation_mask',experiment.m0_proxy>0);
experiment.relaxation.T2star_s = .025+.012*rand(object_size);
nstates = grid_size(1);
rf = .02*(0:nstates-1)';
order = [2:nstates 1]';
line_index = repmat(order,1,grid_size(2));
rf_grid = repmat(rf(order),1,grid_size(2));
[kx,ky] = meshgrid(((0:grid_size(2)-1)-floor(grid_size(2)/2))/.2, ...
    ((0:grid_size(1)-1)-floor(grid_size(1)/2))/.18);
experiment.acquisition = struct('sequence_kind','gre','kx_mInv',kx,'ky_mInv',ky, ...
    'time_s',rf_grid+.004+.0002*(0:grid_size(2)-1), ...
    'rf_time_s',rf_grid,'TE_s',.0045,'line_start_s',rf+.003,'line_end_s',rf+.007, ...
    'line_index',line_index,'phase_offset_rad',.23*line_index.^2);
calibration = struct('X_m',X,'Y_m',Y,'units','Hz/A', ...
    'acdc_maps',80*randn([object_size 32]));
end

function kdata = direct_signal(experiment,calibration,currents)
% Explicit independent oracle: one current state and one RF for every ADC.
acq = experiment.acquisition;
kdata = complex(zeros(size(acq.time_s)));
maps = reshape(calibration.acdc_maps,[],32);
for sample = 1:numel(kdata)
    state = acq.line_index(sample);
    dt = acq.time_s(sample)-acq.rf_time_s(sample);
    shim_duration = acq.time_s(sample)-acq.line_start_s(state);
    phase = experiment.geometry.X_m(:)*acq.kx_mInv(sample) ...
        + experiment.geometry.Y_m(:)*acq.ky_mInv(sample) ...
        + experiment.field.b0Hz(:)*dt+(maps*currents(:,state))*shim_duration;
    decay = exp(-(dt-acq.TE_s)./experiment.relaxation.T2star_s(:));
    kdata(sample) = sum(experiment.m0_proxy(:).*decay ...
        .*exp(-2*pi*1i*phase+1i*acq.phase_offset_rad(sample)));
end
end

function error = relative_error(actual,expected)
error = norm(actual(:)-expected(:))/max(norm(expected(:)),1e-12);
end

function expect_failure(experiment,calibration)
failed = false;
try, prepare_experiment_optimizer(experiment,calibration,.2);
catch, failed = true; end
assert(failed,'Invalid GRE RF/state windows were accepted.');
end
