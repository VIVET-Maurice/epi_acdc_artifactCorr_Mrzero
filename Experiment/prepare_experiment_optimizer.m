function ctx = prepare_experiment_optimizer(experiment, calibration, Imax_A)
%PREPARE_EXPERIMENT_OPTIMIZER Verify a measured basis and cache GRE/EPI physics.
%   CALIBRATION contains acdc_maps [object rows, object columns, 32] in Hz/A, X_m, Y_m and
%   units='Hz/A'. Geometry must already match the notebook's physical grid.
%   IMAX_A is an explicitly supplied hardware limit in amperes (scalar or
%   one positive value per channel). No old-subject maps or surrogate basis
%   are accepted here. The measured coil maps can also be used for a
%   VIRTUAL coil placement: this preparation verifies numerical geometry,
%   not the physical presence or registration of coils during acquisition.

required = {'acdc_maps','X_m','Y_m','units'};
assert(isstruct(calibration) && isscalar(calibration) && ...
    all(isfield(calibration,required)), 'Calibration needs acdc_maps, X_m, Y_m and units.');
assert(strcmp(char(calibration.units),'Hz/A'), 'Calibration units must be explicitly Hz/A.');
m0 = double(experiment.m0_proxy);
validateattributes(m0,{'numeric'},{'real','2d','nonempty','finite','nonnegative'});
object_size = size(m0);
nc = 32;
validateattributes(calibration.acdc_maps,{'numeric'}, ...
    {'real','size',[object_size nc]});
validateattributes(calibration.X_m,{'numeric'},{'real','finite','size',object_size});
validateattributes(calibration.Y_m,{'numeric'},{'real','finite','size',object_size});
validateattributes(experiment.geometry.X_m,{'numeric'},{'real','finite','size',object_size});
validateattributes(experiment.geometry.Y_m,{'numeric'},{'real','finite','size',object_size});
assert(max(abs(calibration.X_m(:)-experiment.geometry.X_m(:)))<=1e-9 && ...
    max(abs(calibration.Y_m(:)-experiment.geometry.Y_m(:)))<=1e-9, ...
    'Calibration coordinates must match the experiment (tolerance 1 nm); register maps first.');
validateattributes(Imax_A,{'numeric'},{'real','finite','vector','positive'});
assert(isscalar(Imax_A) || numel(Imax_A)==nc, 'Imax_A needs one value or 32 channel limits.');
if isscalar(Imax_A), Imax_A = repmat(double(Imax_A),nc,1); end

b0 = double(experiment.field.b0Hz);
validateattributes(m0,{'numeric'},{'real','finite','nonnegative','size',object_size});
validateattributes(b0,{'numeric'},{'real','finite','size',object_size});
assert(isequal(size(experiment.field.mask),object_size) && ...
    isequal(size(experiment.field.simulation_mask),object_size), 'Field masks must match the reference object.');
support = logical(experiment.field.simulation_mask) & m0>0;
quality = logical(experiment.field.mask) & support;
assert(any(support(:)) && any(quality(:)), 'Need object support and reliable measured B0 voxels.');
m0(~support) = 0;
b0(~support) = 0;
maps = reshape(double(calibration.acdc_maps),[],nc);
valid_maps = all(isfinite(maps),2);
if isfield(calibration,'spatial_valid_mask')
    assert(isequal(size(calibration.spatial_valid_mask),object_size), ...
        'Calibration spatial_valid_mask must match the object grid.');
    valid_maps = valid_maps & logical(calibration.spatial_valid_mask(:));
end
assert(all(valid_maps(support(:))), ...
    ['Some simulated object points lack measured ACDC responses. Use the separately ' ...
     'prepared calibration-covered input; unknown responses cannot be set to zero.']);
assert(all(any(abs(maps(support(:),:))>0,1)), ...
    'Every calibrated channel must have a nonzero response within object support.');

acquisition = experiment.acquisition;
validateattributes(acquisition.kx_mInv,{'numeric'},{'real','2d','nonempty','finite'});
grid_size = size(acquisition.kx_mInv);
validateattributes(acquisition.ky_mInv,{'numeric'},{'real','finite','size',grid_size});
validateattributes(acquisition.time_s,{'numeric'},{'real','finite','nonnegative','size',grid_size});
validateattributes(acquisition.TE_s,{'numeric'},{'real','finite','nonnegative','scalar'});
assert(isfield(acquisition,'line_start_s'), ...
    'Export imaging block starts as acquisition.line_start_s (seconds relative to the first RF).');
line_start = double(acquisition.line_start_s(:));
nstates = numel(line_start);
validateattributes(line_start,{'numeric'},{'real','finite','nonnegative','nonempty','column'});
assert(nstates==grid_size(1), 'Each Cartesian row needs one imaging state.');
validateattributes(acquisition.line_index,{'numeric'}, ...
    {'real','finite','integer','>=',1,'<=',nstates,'size',grid_size});
assert(all(all(acquisition.line_index==acquisition.line_index(:,1))) && ...
    numel(unique(acquisition.line_index(:,1)))==nstates, 'Each Cartesian row must identify one distinct acquired line.');
assert(all(diff(line_start)>0), 'Imaging line starts must increase in acquisition order.');
line_end = [line_start(2:end); Inf];
time = double(acquisition.time_s(:));
index = double(acquisition.line_index(:));
sequence_kind = 'epi';
if isfield(acquisition,'sequence_kind')
    sequence_kind = validatestring(lower(char(acquisition.sequence_kind)),{'epi','gre'});
end
% Response parameters are independent of the chosen numerical backend.
% Omitting this configuration keeps the original instantaneous model.
current_response = struct('enabled',false,'tau_s',repmat(133e-6/log(100),nc,1));
if isfield(experiment,'current_response')
    response = experiment.current_response;
    assert(isstruct(response) && isscalar(response), ...
        'experiment.current_response must be a scalar struct.');
    if isfield(response,'enabled')
        flag = response.enabled;
        assert((islogical(flag) || isnumeric(flag)) && isreal(flag) && ...
            isscalar(flag) && any(double(flag)==[0 1]), ...
            'current_response.enabled must be true or false.');
        current_response.enabled = logical(flag);
    end
    if isfield(response,'tau_s')
        validateattributes(response.tau_s,{'numeric'}, ...
            {'real','finite','positive','vector','nonempty'});
        tau = double(response.tau_s(:));
        assert(isscalar(tau) || numel(tau)==nc, ...
            'current_response.tau_s needs one value or 32 channel values in seconds.');
        if isscalar(tau), tau = repmat(tau,nc,1); end
        current_response.tau_s = tau;
    end
end
assert(~current_response.enabled || strcmp(sequence_kind,'epi'), ...
    'The first-order current response is currently supported for EPI only.');
if current_response.enabled && isfield(acquisition,'line_end_s')
    line_end = double(acquisition.line_end_s(:));
    assert(numel(line_end)==nstates && all(~isnan(line_end)) && ...
        all(line_end>line_start) && all(isfinite(line_end(1:end-1))) && ...
        all(line_end(1:end-1)<=line_start(2:end)), ...
        'EPI command windows must not overlap; only the final end may be Inf.');
end
rf_time = zeros(size(time));
if strcmp(sequence_kind,'gre')
    assert(all(isfield(acquisition,{'rf_time_s','line_end_s'})), ...
        'Spoiled GRE requires per-ADC rf_time_s and per-state line_end_s.');
    validateattributes(acquisition.rf_time_s,{'numeric'}, ...
        {'real','finite','nonnegative','size',grid_size});
    rf_time = double(acquisition.rf_time_s(:));
    line_end = double(acquisition.line_end_s(:));
    validateattributes(line_end,{'numeric'},{'real','finite','size',[nstates 1]});
    rf_by_state = zeros(nstates,1);
    for state = 1:nstates
        local_rf = rf_time(index==state);
        assert(max(local_rf)-min(local_rf)<=1e-12, 'A GRE line must have one RF excitation center.');
        rf_by_state(state) = local_rf(1);
    end
    assert(all(diff(rf_by_state)>0) && all(rf_by_state<=line_start) && ...
        all(line_end>line_start) && all(line_end(1:end-1)<=rf_by_state(2:end)), ...
        'GRE state windows must follow their RF and finish before the next RF.');
end
assert(all(time>=line_start(index)-1e-12 & time<line_end(index)), ...
    'Each ADC time must belong to its exported imaging block.');
% H(sample,state) is elapsed time spent at that shim state before that ADC.
% Spoiled GRE restarts the transverse phase at each RF. EPI retains the
% entire history after its single RF, including all earlier imaging states.
H = max(0,min(time,line_end.')-max(rf_time,line_start.'));
evolution_time = time-rf_time;
phase_offset = zeros(size(time));
if isfield(acquisition,'phase_offset_rad')
    validateattributes(acquisition.phase_offset_rad,{'numeric'}, ...
        {'real','finite','size',grid_size});
    phase_offset = double(acquisition.phase_offset_rad(:));
end

T2star = experiment.relaxation.T2star_s;
assert(isnumeric(T2star) && isreal(T2star) && ~isempty(T2star) && ...
    all(~isnan(T2star(:)) & T2star(:)>0), 'T2star must be positive; Inf is allowed.');
if isscalar(T2star), T2star = repmat(double(T2star),object_size); end
assert(isequal(size(T2star),object_size), 'T2star must be scalar or match the reference object.');
amplitude = m0(support);
x = double(experiment.geometry.X_m(support));
y = double(experiment.geometry.Y_m(support));
b0_active = b0(support);
rate = 1./double(T2star(support));
kx = double(acquisition.kx_mInv(:));
ky = double(acquisition.ky_mInv(:));
base = complex(zeros(nnz(support),numel(time)));
reference_signal = complex(zeros(size(time)));
for first = 1:32:numel(time)
    samples = first:min(first+31,numel(time));
    t = evolution_time(samples).';
    transverse = amplitude .* exp(-rate*(t-double(acquisition.TE_s))) ...
        .*exp(1i*phase_offset(samples).');
    spatial = x*kx(samples).' + y*ky(samples).';
    reference_signal(samples) = sum(transverse.*exp(-2*pi*1i*spatial),1).';
    base(:,samples) = transverse.*exp(-2*pi*1i*(spatial+b0_active*t));
end
assert(all(isfinite(base(:))) && all(isfinite(reference_signal)), ...
    'Nonfinite Bloch signal; check echo-relative T2star evolution.');
reference = fftshift(ifft2(ifftshift(reshape(reference_signal,grid_size))));
reference_energy = sum(abs(reference(:)).^2);
assert(reference_energy>0 && isfinite(reference_energy), 'The zero-B0 reference has no usable signal.');
ctx = struct('B',maps(support(:),:),'B_full',maps,'B_quality',maps(quality(:),:), ...
    'base_signal',base,'H_s',H,'nc',nc,'nstates',nstates,'image_size',grid_size, ...
    'object_size',object_size,'support',support,'quality_mask',quality, ...
    'Imax_A',double(Imax_A(:)),'reference',reference,'reference_energy',reference_energy, ...
    'line_start_s',line_start,'line_end_s',line_end,'sequence_kind',sequence_kind, ...
    'current_response',current_response, ...
    'time_s',reshape(time,grid_size),'line_index',double(acquisition.line_index), ...
    'rf_time_s',reshape(rf_time,grid_size),'phase_offset_rad',reshape(phase_offset,grid_size), ...
    'm0_proxy',m0,'b0Hz',b0, ...
    'calibration_valid_mask',reshape(valid_maps,object_size), ...
    'registration_confirmed',isfield(calibration,'registration_confirmed') && ...
        isequal(double(calibration.registration_confirmed),1));
if current_response.enabled
    [ctx.H_response_s,ctx.R_response] = acdc_current_response( ...
        line_start,line_end,time,current_response.tau_s,rf_time);
end
% NaN outside calibrated coverage remains in B_full/output field maps.
% Only finite, covered object points participate in the forward model.
end
