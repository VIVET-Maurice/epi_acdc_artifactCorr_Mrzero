function [image, kdata] = forward_model_experiment(m0, b0Hz, geometry, acquisition, relaxation)
%FORWARD_MODEL_EXPERIMENT Sampled transverse Bloch model for experimental GRE/EPI.
%   M0 is a real effective transverse-magnitude proxy at echo time TE, not a
%   measured proton-density map. Signal evolution therefore uses
%       exp(-(time-TE)/T2star) * exp(-2*pi*1i*(kx*X + ky*Y + B0*time)).
%   Earlier ADC samples can have greater magnitude than this TE proxy.
%   X/Y are physical readout/phase coordinates [m]; kx/ky are [cycles/m].
%   Acquisition arrays and output KDATA have rows=phase and columns=readout,
%   already arranged in nominal Cartesian order; TIME retains true polarity.
%   This models free precession after excitation, one uniform receive coil,
%   and effective T2* decay. It does not model RF/TR steady state, slice
%   profiles, receive-coil sensitivities, diffusion, or intra-voxel dephasing.
%   Navigator phase correction is applied to measured data during export;
%   the ideal simulation has no additional alternating hardware phase error.
%   Spoiled GRE uses time-rf_time_s for each line's independent excitation.
%   Optional phase_offset_rad is RF phase minus receiver/ADC phase.

validateattributes(m0, {'numeric'}, {'real','2d','nonempty','finite','nonnegative'});
object_size = size(m0);
validateattributes(b0Hz, {'numeric'}, {'real','finite','size',object_size});
validateattributes(geometry.X_m, {'numeric'}, {'real','finite','size',object_size});
validateattributes(geometry.Y_m, {'numeric'}, {'real','finite','size',object_size});
validateattributes(acquisition.kx_mInv, {'numeric'}, {'real','2d','nonempty','finite'});
grid_size = size(acquisition.kx_mInv);
validateattributes(acquisition.ky_mInv, {'numeric'}, {'real','finite','size',grid_size});
validateattributes(acquisition.time_s, {'numeric'}, {'real','finite','nonnegative','size',grid_size});
validateattributes(acquisition.TE_s, {'numeric'}, {'real','finite','nonnegative','scalar'});
T2star = relaxation.T2star_s;
assert(isnumeric(T2star) && isreal(T2star) && ~isempty(T2star) ...
    && all(~isnan(T2star(:)) & T2star(:)>0), 'T2star must be positive (Inf is allowed).');
if isscalar(T2star)
    T2star = repmat(double(T2star),object_size);
else
    assert(isequal(size(T2star),object_size), 'T2star must be scalar or size(M0).');
end

% Ignore zero-magnitude voxels and process small ADC blocks to bound memory.
active = m0(:)>0;
amplitude = double(m0(active));
x = double(geometry.X_m(active));
y = double(geometry.Y_m(active));
b0 = double(b0Hz(active));
rate = 1./double(T2star(active));
kx = double(acquisition.kx_mInv(:));
ky = double(acquisition.ky_mInv(:));
time = double(acquisition.time_s(:));
sequence_kind = 'epi';
if isfield(acquisition,'sequence_kind')
    sequence_kind = validatestring(lower(char(acquisition.sequence_kind)),{'epi','gre'});
end
if strcmp(sequence_kind,'gre')
    assert(isfield(acquisition,'rf_time_s'), 'Spoiled GRE requires per-ADC rf_time_s.');
    validateattributes(acquisition.rf_time_s,{'numeric'}, ...
        {'real','finite','nonnegative','size',grid_size});
    time = time-double(acquisition.rf_time_s(:));
    assert(all(time>=0), 'Every GRE ADC sample must follow its RF center.');
end
phase_offset = zeros(size(time));
if isfield(acquisition,'phase_offset_rad')
    validateattributes(acquisition.phase_offset_rad,{'numeric'}, ...
        {'real','finite','size',grid_size});
    phase_offset = double(acquisition.phase_offset_rad(:));
end
signal = complex(zeros(size(time)));
block_size = 32;
for first = 1:block_size:numel(time)
    samples = first:min(first+block_size-1,numel(time));
    t = time(samples).';
    phase_cycles = x*kx(samples).' + y*ky(samples).' + b0*t;
    decay = exp(-rate*(t-double(acquisition.TE_s)));
    signal(samples) = sum(amplitude .* decay .* exp(-2*pi*1i*phase_cycles) ...
        .*exp(1i*phase_offset(samples).'),1).';
end
assert(all(isfinite(signal)), 'Nonfinite signal: inspect T2star and echo-relative decay.');
kdata = reshape(signal,grid_size);
image = fftshift(ifft2(ifftshift(kdata)));
end
