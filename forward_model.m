function [img_hat,kdata] = forward_model( img , B0Hz , shimHzPerAmp , Ishim , params , seq )

Nx = params.matrix_size;
Ny = Nx;
Nc = size(shimHzPerAmp,3);

% Ishim is [Nc Nadc], one shim setting per EPI readout lobe / ky line
Nadc = Ny;

if ~isequal(size(Ishim),[Nc Nadc])
    error('Ishim must be size [Nc Nadc] = [%d %d].',Nc,Nadc);
end

Nt = numel(seq.Gx);

FOV = params.FOV;
dk = 1/FOV;

% Precompute shim field for each ADC event / ky line
[ Nx_  Ny_ ] = size(img);
shimHzByEvent = zeros(Nx_,Ny_,Nadc);

for a = 1:Nadc
    for c = 1:Nc
        shimHzByEvent(:,:,a) = shimHzByEvent(:,:,a) + ...
            shimHzPerAmp(:,:,c) * Ishim(c,a);
    end
end


% Relaxation
T1 = expand_param( params.T1 , Nx_ , Ny_ );
T2 = expand_param( params.T2 , Nx_ , Ny_ );

E1 = exp(-seq.dt ./ T1);
E2 = exp(-seq.dt ./ T2);

% Initial transverse magnetization after ideal 90-degree pulse
M0  = double(img);
Mxy = complex(M0,zeros(Nx_,Ny_));
Mz  = zeros(Nx_,Ny_);

% Centered k-space storage
kdata = zeros(params.matrix_size);

for t = 1 : Nt

    eventIdx = seq.eventOfTime(t);

    % offResHz = B0Hz + shimHzByEvent(:,:,1);

    if eventIdx > 0 
        offResHz = B0Hz + shimHzByEvent(:,:,eventIdx);
    else
        offResHz = B0Hz;
    end

    gradHz = params.gammaHz * (seq.Gx(t)*params.X + seq.Gy(t)*params.Y);
    omegaHz = offResHz + gradHz;

    phi = 2*pi*omegaHz*seq.dt;

    Mxy = Mxy .* exp(-1i*phi) .* E2;
    Mz  = M0 - (M0 - Mz).*E1;

    if seq.adc(t)

        s = sum(Mxy(:));

        kx = seq.kx(t);
        ky = seq.ky(t);

        ix = round(kx/dk) + params.matrix_size/2 + 1;
        iy = round(ky/dk) + params.matrix_size/2 + 1;

        if ix >= 1 && ix <= params.matrix_size && iy >= 1 && iy <= params.matrix_size
            kdata(ix,iy) = s;
        end
    end
end

% Centered Cartesian k-space -> image
img_hat = fftshift(ifft2(ifftshift(kdata)));

end


function A = expand_param(a,Ny,Nx)

if isscalar(a)
    A = a * ones(Ny,Nx);
else
    A = a;
    if ~isequal(size(A),[Ny Nx])
        error('Parameter must be scalar or [Ny Nx].');
    end
end

end