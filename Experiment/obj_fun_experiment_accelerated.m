function [value, gradient] = obj_fun_experiment_accelerated(x, runtime)
%OBJ_FUN_EXPERIMENT_ACCELERATED Exact fused magnitude objective and adjoint.
%   Retains one forward pass, then reuses its complex voxel contributions.
%   The staircase timing permits sample sums and reverse cumulative sums
%   instead of dense H multiplications. Generic timing has an exact fallback.
%   First-order response uses its exact integral and matching adjoint, with
%   independent integral matrices when channel time constants differ.
%   FMINCON receives ordinary CPU doubles even when Device is 'gpu'.
currents_A = reshape(x,runtime.nc,runtime.nstates);
need_gradient = nargout>1;
if need_gradient
    [image,~,contributions] = forward_model_experiment_accelerated(currents_A,runtime);
else
    image = forward_model_experiment_accelerated(currents_A,runtime);
end
amplitude = abs(image);
residual = amplitude-runtime.reference_amplitude;
value = sum(residual(:).^2)/runtime.reference_energy;
if need_gradient
    lambda_image = complex(zeros(runtime.image_size,'like',amplitude));
    nonzero = amplitude>0;
    lambda_image(nonzero) = 2*residual(nonzero).*image(nonzero) ...
        ./amplitude(nonzero)/runtime.reference_energy;
    lambda_signal = fftshift(fft2(ifftshift(lambda_image)))/prod(runtime.image_size);
    lambda_signal = lambda_signal(:);
    structured = runtime.acceleration.TimingStructured;
    channel_response = isfield(runtime.acceleration,'ResponseEnabled') && ...
        runtime.acceleration.ResponseEnabled && ~runtime.acceleration.ResponseCommonTau;
    if structured
        state_sum = zeros(size(runtime.B,1),runtime.nstates,'like',runtime.B);
        state_weighted = zeros(size(state_sum),'like',state_sum);
    else
        gradient_I = zeros(runtime.nc,runtime.nstates,'like',runtime.B);
    end
    for block = 1:numel(runtime.blocks)
        b = runtime.blocks(block);
        sensitivity = 2*pi*imag(contributions{block} ...
            .*conj(lambda_signal(b.OriginalSamples).'));
        if channel_response
            projected = runtime.BT*sensitivity;
            for channel = 1:runtime.nc
                gradient_I(channel,:) = gradient_I(channel,:) ...
                    + projected(channel,:)*b.H_response_s(:,:,channel);
            end
        elseif structured
            for group = 1:numel(b.UniqueStates)
                state = b.UniqueStates(group);
                selected = b.StateColumns{group};
                selected_sensitivity = sensitivity(:,selected);
                state_sum(:,state) = state_sum(:,state)+sum(selected_sensitivity,2);
                state_weighted(:,state) = state_weighted(:,state) ...
                    + selected_sensitivity*b.Elapsed_s(selected).';
            end
        else
            % Project voxels into the small coil basis before multiplying H.
            gradient_I = gradient_I + (runtime.BT*sensitivity)*b.H_s;
        end
    end
    if structured
        later_states = [fliplr(cumsum(fliplr(state_sum(:,2:end)),2)), ...
            zeros(size(state_sum,1),1,'like',state_sum)];
        field_gradient = state_weighted+later_states.*runtime.state_duration_s;
        gradient_I = runtime.BT*field_gradient;
    end
    gradient = gradient_I(:);
    if isa(gradient,'gpuArray'), gradient = gather(gradient); end
end
if isa(value,'gpuArray'), value = gather(value); end
end
