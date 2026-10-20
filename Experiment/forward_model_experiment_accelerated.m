function [image, kdata, contributions] = forward_model_experiment_accelerated(currents_A, runtime)
%FORWARD_MODEL_EXPERIMENT_ACCELERATED Exact forward pass with optional cache.
%   A third output retains contributions for one adjoint, avoiding a second
%   phase/exponential evaluation. Outputs remain on the chosen CPU/GPU device.
assert(isreal(currents_A) && isequal(size(currents_A),[runtime.nc runtime.nstates]), ...
    'Currents must be real channels-by-states.');
if strcmp(runtime.acceleration.Device,'gpu') && ~isa(currents_A,'gpuArray')
    currents_A = gpuArray(double(currents_A));
end
fields = runtime.B*double(currents_A);
structured = runtime.acceleration.TimingStructured;
channel_response = isfield(runtime.acceleration,'ResponseEnabled') && ...
    runtime.acceleration.ResponseEnabled && ~runtime.acceleration.ResponseCommonTau;
if structured
    phase_start = [zeros(size(fields,1),1,'like',fields), ...
        cumsum(fields(:,1:end-1).*runtime.state_duration_s(1:end-1),2)];
end
signal = complex(zeros(size(runtime.H_s,1),1,'like',fields));
retain = nargout>=3;
if retain, contributions = cell(size(runtime.blocks)); end
for block = 1:numel(runtime.blocks)
    b = runtime.blocks(block);
    if channel_response
        % Integrate each real channel current since RF, then project its
        % accumulated phase through the measured Hz/A coil basis.
        integrated_currents = zeros(runtime.nc,numel(b.OriginalSamples),'like',fields);
        for channel = 1:runtime.nc
            integrated_currents(channel,:) = double(currents_A(channel,:)) ...
                * b.H_response_s(:,:,channel).';
        end
        phase = runtime.B*integrated_currents;
    elseif structured
        phase = phase_start(:,b.StateIndex) + fields(:,b.StateIndex).*b.Elapsed_s;
    else
        phase = fields*b.H_s.';
    end
    contribution = b.BaseSignal.*exp(-2*pi*1i*phase);
    signal(b.OriginalSamples) = sum(contribution,1).';
    if retain, contributions{block} = contribution; end
end
kdata = reshape(signal,runtime.image_size);
image = fftshift(ifft2(ifftshift(kdata)));
end
