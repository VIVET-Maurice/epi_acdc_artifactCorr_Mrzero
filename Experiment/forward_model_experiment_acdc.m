function [image, kdata, gradient] = forward_model_experiment_acdc(currents_A, ctx, lambda_image)
%FORWARD_MODEL_EXPERIMENT_ACDC Exact sampled Bloch signal with accumulated shim phase.
%   CURRENTS_A is [32 channels x number of imaging lines]. The cache holds the true
%   Cartesian ADC coordinates/times, not the legacy simulation's gridding.
%   An optional image adjoint LAMBDA_IMAGE computes d(Re<lambda,image>)/dI.
%   All shim currents are zero before the first imaging block; this includes
%   EPI navigators. GRE histories restart at each spoiled RF excitation.
%   When ctx.current_response.enabled is true, each channel follows its
%   first-order response. Phase uses its exact current integral, including
%   residual current from previous states. Otherwise changes are immediate.

validateattributes(currents_A,{'numeric'}, ...
    {'real','finite','size',[ctx.nc ctx.nstates]});
use_adjoint = nargin>=3;
use_response = isfield(ctx,'current_response') && ctx.current_response.enabled;
% A shared tau gives one integration matrix for all channels. Retain the
% faster field-by-state calculation in this usual notebook configuration.
per_channel_response = use_response && ...
    any(ctx.current_response.tau_s~=ctx.current_response.tau_s(1));
if use_adjoint
    validateattributes(lambda_image,{'numeric'},{'finite','size',ctx.image_size});
    lambda_signal = fftshift(fft2(ifftshift(lambda_image)))/prod(ctx.image_size);
    lambda_signal = lambda_signal(:);
    if per_channel_response
        gradient = zeros(size(currents_A));
    else
        field_gradient = zeros(size(ctx.B,1),ctx.nstates);
    end
end
if ~per_channel_response, fields = ctx.B*double(currents_A); end
signal = complex(zeros(size(ctx.H_s,1),1));
for first = 1:32:numel(signal)
    samples = first:min(first+31,numel(signal));
    if per_channel_response
        integrated_current = zeros(ctx.nc,numel(samples));
        for channel = 1:ctx.nc
            integrated_current(channel,:) = double(currents_A(channel,:)) ...
                *ctx.H_response_s(samples,:,channel).';
        end
        phase_cycles = ctx.B*integrated_current;
    else
        if use_response
            H = ctx.H_response_s(samples,:,1);
        else
            H = ctx.H_s(samples,:);
        end
        phase_cycles = fields*H.';
    end
    contribution = ctx.base_signal(:,samples).*exp(-2*pi*1i*phase_cycles);
    signal(samples) = sum(contribution,1).';
    if use_adjoint
        sensitivity = 2*pi*imag(contribution.*conj(lambda_signal(samples).'));
        if per_channel_response
            channel_sensitivity = ctx.B.'*sensitivity;
            for channel = 1:ctx.nc
                gradient(channel,:) = gradient(channel,:) ...
                    + channel_sensitivity(channel,:)*ctx.H_response_s(samples,:,channel);
            end
        else
            field_gradient = field_gradient + sensitivity*H;
        end
    end
end
kdata = reshape(signal,ctx.image_size);
image = fftshift(ifft2(ifftshift(kdata)));
if use_adjoint && ~per_channel_response, gradient = ctx.B.'*field_gradient; end
end
