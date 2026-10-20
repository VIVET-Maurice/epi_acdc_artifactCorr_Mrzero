function [actual_A, commanded_A, times_s] = acdc_sample_current_waveform(currents_A, starts_s, ends_s, times_s, current_response)
%ACDC_SAMPLE_CURRENT_WAVEFORM Sample commands and their first-order response.
%   Columns in CURRENTS_A command each channel during [STARTS_S, ENDS_S).
%   Output columns follow TIMES_S, including unsorted or repeated times.
%   Commands are zero outside the windows. Actual currents start at zero,
%   stay continuous at every boundary, and decay after a finite final end.
%   CURRENT_RESPONSE has enabled and tau_s (scalar or one value per channel).
%   Empty TIMES_S creates a full plotting grid around every transition.
%   A finite final end adds a settling tail. With a final Inf window, the
%   automatic grid stops one typical state duration after the final start.
%
%   This uses an exact recurrence between event/sample times. It preserves
%   all earlier transitions without allocating sample-by-state-by-channel
%   response weights, so a full waveform can be plotted at fine resolution.

validateattributes(currents_A,{'numeric'},{'real','finite','2d','nonempty'});
validateattributes(starts_s,{'numeric'}, ...
    {'real','finite','nonnegative','vector','nonempty'});
validateattributes(ends_s,{'numeric'},{'real','vector','nonempty'});
commands = double(currents_A);
starts = double(starts_s(:));
ends = double(ends_s(:));
nc = size(commands,1);
assert(size(commands,2)==numel(starts) && numel(ends)==numel(starts), ...
    'Each current state needs one start and one end.');
assert(all(diff(starts)>0) && all(~isnan(ends)) && all(ends>starts) && ...
    all(isfinite(ends(1:end-1))) && all(ends(1:end-1)<=starts(2:end)), ...
    'Current windows must be ordered and nonoverlapping; only the final end may be Inf.');
assert(isstruct(current_response) && isscalar(current_response) && ...
    isfield(current_response,'enabled') && isscalar(current_response.enabled), ...
    'current_response needs a scalar enabled flag.');
flag = current_response.enabled;
assert((islogical(flag) || isnumeric(flag)) && isreal(flag) && ...
    any(double(flag)==[0 1]), 'current_response.enabled must be true or false.');
validateattributes(current_response.enabled,{'logical','numeric'}, ...
    {'real','finite','scalar'});
assert(current_response.enabled==0 || current_response.enabled==1, ...
    'current_response.enabled must be true or false.');
tau = [];
if current_response.enabled
    assert(isfield(current_response,'tau_s'),'Enabled response needs tau_s.');
    validateattributes(current_response.tau_s,{'numeric'}, ...
        {'real','finite','positive','vector','nonempty'});
    tau = double(current_response.tau_s(:));
    if isscalar(tau), tau = repmat(tau,nc,1); end
    assert(numel(tau)==nc,'tau_s must be scalar or contain one value per channel.');
end
if isempty(times_s)
    times_s = plotting_times(starts,ends,tau);
end
validateattributes(times_s,{'numeric'}, ...
    {'real','finite','nonnegative','vector','nonempty'});
times_s = double(times_s(:));
times = times_s;
if ~current_response.enabled
    commanded_A = commands*double(times>=starts.' & times<ends.').';
    actual_A = commanded_A;
    return
end

% Every boundary changes to a new state, or to zero at the end of a window.
finite_end = isfinite(ends);
[event_times,~,event_index] = unique([starts;ends(finite_end)]);
event_commands = zeros(nc,numel(event_times));
for state = 1:numel(starts)
    column = event_index(state);
    event_commands(:,column) = commands(:,state);
end

% Include every preceding event, even when requested samples start later.
timeline = unique([0;times;event_times(event_times<=max(times))]);
[~,sample_index] = ismember(times,timeline);
actual_timeline = zeros(nc,numel(timeline));
command_timeline = zeros(size(actual_timeline));
actual = zeros(nc,1);
command = zeros(nc,1);
previous_time = 0;
event = 1;
for sample = 1:numel(timeline)
    time = timeline(sample);
    % The command held over the preceding interval determines the response.
    actual = command+(actual-command).*exp(-(time-previous_time)./tau);
    if event<=numel(event_times) && event_times(event)==time
        command = event_commands(:,event);
        event = event+1;
    end
    actual_timeline(:,sample) = actual;
    command_timeline(:,sample) = command;
    previous_time = time;
end
actual_A = actual_timeline(:,sample_index);
commanded_A = command_timeline(:,sample_index);
end

function times = plotting_times(starts,ends,tau)
% Resolve all rising/falling edges without oversampling long plateaus.
boundaries = unique([starts;ends(isfinite(ends))]);
if isfinite(ends(end))
    final_time = ends(end);
else
    durations = [diff(starts);ends(isfinite(ends))-starts(isfinite(ends))];
    if isempty(durations)
        assert(~isempty(tau), ...
            'Specify sample times for an instantaneous single state with no finite end.');
        durations = 8*max(tau);
    end
    final_time = starts(end)+median(durations);
end
times = [0;boundaries;final_time];
if ~isempty(tau)
    % Distinct channel time constants get their own grid near each edge.
    offsets = unique(reshape(linspace(0,8,81).'*unique(tau).',[],1));
    times = [times;reshape(boundaries+offsets.',[],1)];
    if isfinite(ends(end)), final_time = final_time+8*max(tau); end
end
% A point just before each edge makes the overlaid DAC command step visible.
epsilon = min(1e-9,max(final_time,eps)*1e-6);
times = unique([times;max(0,boundaries-epsilon);final_time]);
times = times(times<=final_time);
end
