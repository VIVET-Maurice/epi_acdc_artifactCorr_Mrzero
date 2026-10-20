

function [Ishim,results] = initialize_acdc_currents( B0Hz , shimHzPerAmp , Imin , Imax , mask , display )
% initialize_acdc_currents
%
% Solves:
%
%   min_x || B0 + S*x ||_2^2
%
% where:
%
%   B0Hz          : [Ny Nx] field map in Hz
%   shimHzPerAmp  : [Ny Nx Nc] ACDC basis fields in Hz/A
%   Imin          : [Nc x 1] minimum current (A)
%   Imax          : [Nc x 1] maximum current (A)
%   mask          : [Ny Nx] logical fitting mask
%
% Returns:
%
%   Ishim         : [Nc x 1] optimized currents (A)
%   results       : structure with diagnostics

[Ny,Nx,Nc] = size(shimHzPerAmp);

if nargin < 5 || isempty(mask)
    mask = true(Ny,Nx);
end

mask = logical(mask);

%% Build least-squares system
b = -B0Hz(mask);
Nvox = numel(b);

A = zeros(Nvox,Nc);

for c = 1:Nc
    tmp = shimHzPerAmp(:,:,c);
    A(:,c) = tmp(mask);
end

%% Quadratic program
%
% min ||A*x - b||^2
%
% = min 0.5*x'Hx + f'x

H = 2*(A'*A);
f = -2*(A'*b);

% Numerical regularization
lambda = 1e-8 * trace(H)/Nc;

if ~isfinite(lambda) || lambda <= 0
    lambda = 1e-8;
end

H = H + lambda*eye(Nc);

opts = optimoptions( ...
    'quadprog', ...
    'Algorithm','interior-point-convex', ...
    'Display','off');

I1 = ones(Nc,1)* Imin;
I2 = ones(Nc,1)* Imax;

[Ishim,fval,exitflag,output] = quadprog( ...
    H,f, ...
    [],[],[],[], ...
    I1,I2, ...
    [], ...
    opts);

%% Compute shimmed B0
shimFieldHz = zeros(Ny,Nx);

for c = 1:Nc
    shimFieldHz = shimFieldHz + ...
        shimHzPerAmp(:,:,c)  * Ishim(c);
end

B0ResidualHz = B0Hz + shimFieldHz;

%% Diagnostics
rmsBefore = sqrt(mean(B0Hz(mask).^2));
rmsAfter  = sqrt(mean(B0ResidualHz(mask).^2));

results.fval = fval;
results.exitflag = exitflag;
results.output = output;

results.shimFieldHz = shimFieldHz;
results.B0ResidualHz = B0ResidualHz;

results.rmsBeforeHz = rmsBefore;
results.rmsAfterHz = rmsAfter;

results.percentImprovement = ...
    100*(1 - rmsAfter/rmsBefore);

if display
    figure; 
    subplot( 2 , 2 , 1 ); imagesc( B0Hz ); axis image off; colorbar; caxis([-200 200]); title('2SH');
    subplot( 2 , 2 , 2 ); imagesc( B0ResidualHz ); axis image off; colorbar; caxis([-200 200]); title('2SH + ACDC');
    subplot( 2 , 1 , 2 ); 
    H1 = histogram( B0Hz(mask) ); hold on;
    H1.FaceColor = 'red';
    H1.FaceAlpha = 0.5;
    H2 = histogram( B0ResidualHz(mask) );
    H2.FaceColor = 'blue';
    H2.FaceAlpha = 0.5;
    box off;
    xlabel( '\DeltaB_0 [Hz]' );
    ylabel( '# of pixels' );
    sig1 = std( B0Hz(find(mask)) );
    sig2 = std( B0ResidualHz(find(mask)) );
    legend( sprintf( '2SH (\\sigma_{B0}=%3.2f Hz)' , sig1 ) , sprintf( '2SH + ACDC (\\sigma_{B0}=%3.2f Hz)' , sig2 ) , ...
        'Location' , 'best' );
end

end

