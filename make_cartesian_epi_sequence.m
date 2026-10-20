function seq = make_cartesian_epi_sequence( params , display)

dt = params.dt;
Nx = params.matrix_size;
Ny = Nx;
Gmax = params.Gmax;
Smax = params.Smax;
gammaHz = 42.57747892e6;

dk = 1/params.FOV(1);

% Readout design
Gread = params.Gmax;
adc_dt = dk/(gammaHz*Gread);
adc_step = ceil(adc_dt/dt);
adc_dt = adc_step*dt;
Gread = dk/(gammaHz*adc_dt);


% Ramp to first readout lobe
gx0 = slew_ramp(0,Gread,params.Smax,dt);

first_adc_idx = round(adc_step/2);

kx_first = -((Nx-1)/2) * dk;
ky_first = -((Ny-1)/2) * dk;

% Gx prephaser includes compensation for ramp to first readout
% and time between flat-top start and first ADC sample
kx_pre_area = ...
    kx_first/gammaHz ...
    - sum(gx0)*dt ...
    - Gread*first_adc_idx*dt;

ky_pre_area = ky_first/gammaHz;

gx_pre = make_slew_limited_area_pulse(kx_pre_area,Gmax,Smax,dt);
gy_pre = make_slew_limited_area_pulse(ky_pre_area,Gmax,Smax,dt);

n_pre = max(numel(gx_pre),numel(gy_pre));
gx_pre = pad_with_zeros(gx_pre,n_pre);
gy_pre = pad_with_zeros(gy_pre,n_pre);

Gx  = gx_pre;
Gy  = gy_pre;
adc = false(n_pre,1);
eventOfTime = zeros(n_pre,1);   % 0 = prewinder / no ACDC event

kx_index = [];
ky_index = [];

% First Gx lobe begins here: event 1
Gx  = [Gx; gx0];
Gy  = [Gy; zeros(size(gx0))];
adc = [adc; false(size(gx0))];
eventOfTime = [eventOfTime; ones(size(gx0))];

% EPI train
for ky = 1:Ny

    polarity = (-1)^(ky-1);

    gx_flat = polarity * Gread * ones(Nx*adc_step,1);
    gy_flat = zeros(size(gx_flat));

    Gx = [Gx; gx_flat];
    Gy = [Gy; gy_flat];

    adc_flat = false(size(gx_flat));

    for ix = 1:Nx
        sample_idx = (ix-1)*adc_step + first_adc_idx;
        adc_flat(sample_idx) = true;

        if polarity > 0
            kx_index(end+1,1) = ix;
        else
            kx_index(end+1,1) = Nx - ix + 1;
        end

        ky_index(end+1,1) = ky;
    end

    adc = [adc; adc_flat];
    eventOfTime = [eventOfTime; ky*ones(size(gx_flat))];

    % Transition to next lobe
    if ky < Ny
        [gx_trans,gy_trans,zero_idx] = make_epi_transition( ...
            polarity*Gread, ...
           -polarity*Gread, ...
            dk/gammaHz, ...
            Gmax,Smax,dt);

        ev_trans = ky*ones(size(gx_trans));
        ev_trans(zero_idx:end) = ky + 1;

        Gx  = [Gx; gx_trans];
        Gy  = [Gy; gy_trans];
        adc = [adc; false(size(gx_trans))];
        eventOfTime = [eventOfTime; ev_trans];
    end
end


% Ramp final readout gradient back to zero
final_polarity = (-1)^(Ny-1);
gx_end = slew_ramp(final_polarity*Gread,0,Smax,dt);

Gx  = [Gx; gx_end];
Gy  = [Gy; zeros(size(gx_end))];
adc = [adc; false(size(gx_end))];
eventOfTime = [eventOfTime; Ny*ones(size(gx_end))];

% Store sequence
seq.Gx = Gx;
seq.Gy = Gy;
seq.adc = logical(adc);
seq.eventOfTime = eventOfTime;

seq.kx_index = kx_index;
seq.ky_index = ky_index;

seq.dt = dt;
seq.dk = dk;
seq.adc_step = adc_step;
seq.Gread = Gread;

% Continuous k-space trajectory, cycles/m
seq.kx = gammaHz * cumsum(Gx) * dt;
seq.ky = gammaHz * cumsum(Gy) * dt;

seq.kx_adc = seq.kx(seq.adc);
seq.ky_adc = seq.ky(seq.adc);

check_grad_limits(Gx,Gmax,Smax,dt,'Gx');
check_grad_limits(Gy,Gmax,Smax,dt,'Gy');

fprintf( 'EPI protocol:\n' );
fprintf( '   sampling time = %f us\n' , params.dt*1e6 );
fprintf( '   readout BW    = %f Hz/pixel\n' , 1/params.dt/params.matrix_size(1) );
esp = ( numel(gx0)*2 + Nx*adc_step ) * dt;
fprintf( '    ESP           = %f ms\n' , esp*1000 );


if display 
    Nt = numel(seq.Gx);
    time = [ 1 : Nt ] * params.dt;
    figure; 
    subplot(2,1,1); hold on;
    plot(time*1000 , seq.Gx*1000 , '-b');
    plot(time*1000 , seq.Gy*1000 , '-r');
    ind = find(seq.adc>0);
    plot(time(ind)*1000 , seq.adc(ind)*1100*max(abs([seq.Gx])) , 'xk');
    % colors = 'rbgkcmy';
    % for ii = 1 : numel(seq.c)
    %     adc = seq.eventOfTime(ii);
    %     if adc > 0
    %         plot(time(ii)*1000 , 1200*max(abs([seq.Gx])) , 'x' , 'Color' , colors(mod(adc-1,numel(colors))+1));
    %     end
    % end
    box off; title('Gx/Gy [mT/m]'); 
    
    subplot(2,1,2); hold on;
    plot(time(1:end-1)*1000 , diff(seq.Gx)/params.dt , '-b');
    plot(time(1:end-1)*1000 , diff(seq.Gy)/params.dt , '-r');
    box off; title('Sx/Sy [T/m/s]');  xlabel('Time [ms]');
    
    figure;
    plot( seq.kx , seq.ky , '-k' ); box off; axis image;
    xlabel('kx [1/m]');
    ylabel('ky [1/m]');
end

end


function g = slew_ramp(g0,g1,Smax,dt)

dGmax = Smax*dt;
n = ceil(abs(g1-g0)/dGmax);

if n == 0
    g = [];
    return
end

g = linspace(g0,g1,n+1).';
g = g(2:end);

end


function [gx,gy,zero_idx] = make_epi_transition(Gx0,Gx1,gy_area,Gmax,Smax,dt)

gx = slew_ramp(Gx0,Gx1,Smax,dt);
gy_blip = make_slew_limited_area_pulse(gy_area,Gmax,Smax,dt);

[~,zero_idx] = min(abs(gx));

n_gy = numel(gy_blip);
gy_start = zero_idx - floor(n_gy/2);
gy_end   = gy_start + n_gy - 1;

if gy_start < 1
    prepend = 1 - gy_start;
    gx = [Gx0*ones(prepend,1); gx];
    zero_idx = zero_idx + prepend;
    gy_start = 1;
    gy_end = gy_start + n_gy - 1;
end

if gy_end > numel(gx)
    append = gy_end - numel(gx);
    gx = [gx; Gx1*ones(append,1)];
end

gy = zeros(size(gx));
gy(gy_start:gy_end) = gy_blip;

end


function g = make_slew_limited_area_pulse(area,Gmax,Smax,dt)

sgn = sign(area);
A = abs(area);

if A == 0
    g = 0;
    return
end

dGmax = Smax*dt;

n_slew = ceil(sqrt(A/(dGmax*dt)));
n_amp  = ceil(A/(Gmax*dt));

n = max([1,n_slew,n_amp]);

shape = [0:n, n-1:-1:1, 0].';
dG = A/(sum(shape)*dt);

g = sgn * shape * dG;

check_grad_limits(g,Gmax,Smax,dt,'area pulse');

end


function g = pad_with_zeros(g,n)

if numel(g) < n
    g = [g; zeros(n-numel(g),1)];
end

end


function check_grad_limits(g,Gmax,Smax,dt,name)

if isempty(g)
    return
end

if max(abs(g)) > Gmax*(1+1e-12)
    error('%s exceeds Gmax.',name);
end

if max(abs(diff(g)))/dt > Smax*(1+1e-12)
    error('%s exceeds Smax.',name);
end

end