function [H_s, R] = acdc_current_response(line_start_s, line_end_s, times_s, tau_s, rf_time_s)
%ACDC_CURRENT_RESPONSE Exact first-order response to rectangular DAC states.
%   Each state commands a constant current on [LINE_START_S, LINE_END_S).
%   Commands are zero outside these windows. Actual current starts at zero
%   before the first state and remains continuous at every transition.
%   TAU_S is positive seconds: scalar or one value per channel. A measured
%   0-to-1 A step reaching 99 percent in 133 us gives tau=133e-6/log(100).
%
%   R(sample,state,channel) maps commanded amperes to actual ADC amperes.
%   H_S(sample,state,channel) maps commanded amperes to integrated current
%   [A*s] between RF_TIME_S and TIMES_S. RF_TIME_S defaults to zero for EPI.
%   Thus both field and accumulated phase include all earlier transitions.
%   This linear approximation needs experimental validation per channel.

validateattributes(line_start_s,{'numeric'}, ...
    {'real','finite','nonnegative','vector','nonempty'});
validateattributes(line_end_s,{'numeric'},{'real','vector','nonempty'});
validateattributes(times_s,{'numeric'},{'real','finite','nonnegative','vector','nonempty'});
validateattributes(tau_s,{'numeric'},{'real','finite','positive','vector','nonempty'});
starts = double(line_start_s(:).');
ends = double(line_end_s(:).');
times = double(times_s(:));
tau = double(tau_s(:));
assert(numel(ends)==numel(starts) && all(diff(starts)>0), ...
    'Response windows need one end per strictly increasing start.');
assert(all(~isnan(ends)) && all(ends>starts) && ...
    all(isfinite(ends(1:end-1))) && all(ends(1:end-1)<=starts(2:end)), ...
    'Response windows must not overlap; only the final end may be Inf.');
if nargin<5
    rf_time_s = zeros(size(times));
end
validateattributes(rf_time_s,{'numeric'},{'real','finite','nonnegative','vector'});
if isscalar(rf_time_s), rf_time_s = repmat(double(rf_time_s),size(times)); end
rf_time = double(rf_time_s(:));
assert(isequal(size(rf_time),size(times)) && all(rf_time<=times), ...
    'Each ADC needs an RF center no later than its acquisition time.');

H_s = zeros(numel(times),numel(starts),numel(tau));
if nargout>1, R = zeros(size(H_s)); end
for channel = 1:numel(tau)
    % A rectangular command is the difference of two unit steps. Each
    % response tail persists after the next command rather than resetting.
    start_elapsed = max(0,times-starts);
    end_elapsed = max(0,times-ends);
    H_s(:,:,channel) = step_integral(start_elapsed,tau(channel)) ...
        - step_integral(end_elapsed,tau(channel)) ...
        - step_integral(max(0,rf_time-starts),tau(channel)) ...
        + step_integral(max(0,rf_time-ends),tau(channel));
    if nargout>1
        R(:,:,channel) = -expm1(-start_elapsed/tau(channel)) ...
            + expm1(-end_elapsed/tau(channel));
    end
end
end

function integral_s = step_integral(elapsed_s,tau_s)
% Integral of 1-exp(-t/tau), evaluated stably for very short intervals.
x = elapsed_s/tau_s;
integral_s = elapsed_s+tau_s*expm1(-x);
small = x<1e-4;
z = x(small);
integral_s(small) = tau_s*z.^2.*(.5-z/6+z.^2/24-z.^3/120);
end
