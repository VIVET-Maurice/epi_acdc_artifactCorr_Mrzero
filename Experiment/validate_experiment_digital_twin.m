function checks = validate_experiment_digital_twin()
%VALIDATE_EXPERIMENT_DIGITAL_TWIN Analytic physics/orientation checks, no scans.
N = 8;
FOV = 0.2;
position = (-N/2:N/2-1)*FOV/N;
[X,Y] = meshgrid(position,position);
frequency = (-N/2:N/2-1)/FOV;
[kx,ky] = meshgrid(frequency,frequency);
geometry = struct('X_m',X,'Y_m',Y);
acquisition = struct('kx_mInv',kx,'ky_mInv',ky, ...
    'time_s',0.012+(0:N*N-1)'*11e-6,'TE_s',0.012);
acquisition.time_s = reshape(acquisition.time_s,N,N);
relaxation = struct('T2star_s',Inf);

% On a matching spatial/Cartesian grid the zero-field Bloch model equals FFT.
m0 = reshape(1:N*N,N,N)/N^2;
[image,kdata] = forward_model_experiment(m0,zeros(N),geometry,acquisition,relaxation);
reference_kdata = fftshift(fft2(ifftshift(m0)));
checks.zero_field_k_relative_error = norm(kdata(:)-reference_kdata(:))/norm(reference_kdata(:));
checks.zero_field_image_relative_error = norm(image(:)-m0(:))/norm(m0(:));
assert(checks.zero_field_k_relative_error<1e-12,'Zero-B0 Fourier check failed.');
assert(checks.zero_field_image_relative_error<1e-12,'Zero-B0 image check failed.');

% A single source has an exact known phase and echo-relative decay at each ADC.
m0 = zeros(N); row = 3; column = 6; m0(row,column) = 2;
b0 = zeros(N); b0(row,column) = 73;
relaxation.T2star_s = 0.025;
acquisition.time_s = acquisition.time_s-0.0004; % Include pre-TE ADC samples.
[~,kdata] = forward_model_experiment(m0,b0,geometry,acquisition,relaxation);
expected = 2*exp(-(acquisition.time_s-acquisition.TE_s)/0.025) .* ...
    exp(-2*pi*1i*(kx*X(row,column)+ky*Y(row,column)+73*acquisition.time_s));
checks.point_source_max_error = max(abs(kdata(:)-expected(:)));
assert(checks.point_source_max_error<1e-12,'Off-resonance point-source check failed.');

% Columns represent readout X, rows phase Y: neither axis is transposed.
relaxation.T2star_s = Inf;
image = forward_model_experiment(m0,zeros(N),geometry,acquisition,relaxation);
[~,maximum] = max(abs(image(:)));
[peak_row,peak_column] = ind2sub([N N],maximum);
assert(peak_row==row && peak_column==column,'Readout/phase orientation is incorrect.');
checks.peak_row = peak_row;
checks.peak_column = peak_column;

% The derivative of signal with respect to a uniform field is analytic.
step_Hz = 1e-4;
[~,kp] = forward_model_experiment(m0,b0+step_Hz,geometry,acquisition,relaxation);
[~,km] = forward_model_experiment(m0,b0-step_Hz,geometry,acquisition,relaxation);
[~,k0] = forward_model_experiment(m0,b0,geometry,acquisition,relaxation);
numerical = (kp-km)/(2*step_Hz);
analytic = -2*pi*1i*acquisition.time_s.*k0;
checks.field_derivative_relative_error = norm(numerical(:)-analytic(:))/norm(analytic(:));
assert(checks.field_derivative_relative_error<1e-7,'Field-phase derivative check failed.');
checks.passed = true;
fprintf('Experimental Bloch validation passed: FFT %.3g; point %.3g; derivative %.3g.\n', ...
    checks.zero_field_image_relative_error,checks.point_source_max_error,checks.field_derivative_relative_error);
end
