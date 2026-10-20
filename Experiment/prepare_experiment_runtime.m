function runtime = prepare_experiment_runtime(ctx, varargin)
%PREPARE_EXPERIMENT_RUNTIME Cache an exact CPU/GPU execution plan.
%   RUNTIME = prepare_experiment_runtime(CTX,'Device','cpu','BlockSize',256)
%   keeps the physics, units, objective and double precision of CTX.
%   GPU execution uses one GPU in the calling process; there is no reset.
%   BlockSize defaults to 256 samples on CPU and all samples on GPU.
%   The original CTX is unchanged. Use the *_accelerated functions with the
%   returned runtime; its base_signal is replaced by contiguous block caches.
%   First-order response integrals are cached on the selected device. Equal
%   channel time constants share one matrix; different constants stay separate.

parser = inputParser;
parser.PartialMatching = false;
addParameter(parser,'Device','cpu',@(v) (ischar(v) && isrow(v)) || ...
    (isstring(v) && isscalar(v)));
addParameter(parser,'BlockSize',[],@(v) isempty(v) || (isnumeric(v) && ...
    isreal(v) && isscalar(v) && isfinite(v) && v==fix(v) && v>=1));
parse(parser,varargin{:});
device = validatestring(lower(char(parser.Results.Device)),{'cpu','gpu'});
nsamples = size(ctx.H_s,1);
nstates = ctx.nstates;
assert(size(ctx.H_s,2)==nstates && size(ctx.base_signal,2)==nsamples && ...
    size(ctx.base_signal,1)==size(ctx.B,1), 'Inconsistent prepared cache dimensions.');
block_size = parser.Results.BlockSize;
if isempty(block_size)
    if strcmp(device,'gpu'), block_size = nsamples; else, block_size = 256; end
end
block_size = min(double(block_size),nsamples);

% H(sample,state) integrates elapsed time at each piecewise-constant state.
% All earlier states have their entire duration; one state can be partial.
% Verify this structure explicitly before using cumulative phase/reductions.
durations = diff(double(ctx.line_start_s(:))).';
assert(numel(durations)==nstates-1 && all(durations>0), 'Invalid imaging state starts.');
H = double(ctx.H_s);
response_enabled = isfield(ctx,'current_response') && ctx.current_response.enabled;
common_tau = false;
if response_enabled
    assert(isfield(ctx,'H_response_s') && ...
        isequal(size(ctx.H_response_s),[nsamples nstates ctx.nc]), ...
        'The first-order response cache is missing or inconsistent.');
    common_tau = all(ctx.current_response.tau_s==ctx.current_response.tau_s(1));
end
% The last contributing state also identifies GRE rows correctly, where all
% older RF histories have vanished. Those rows use the generic H path below.
state_index = max((H>0).*(1:nstates),[],2);
state_index = max(1,state_index); % all-zero rows belong to the first state
elapsed = H(sub2ind(size(H),(1:nsamples).',state_index));
predicted = zeros(size(H));
for state = 1:nstates
    at_state = state_index==state;
    if state>1, predicted(at_state,1:state-1) = repmat(durations(1:state-1),nnz(at_state),1); end
    predicted(at_state,state) = elapsed(at_state);
end
structured = max(abs(H(:)-predicted(:)))<=1e-14;
% A first-order response has continuous tails, rather than staircase timing.
% Equal time constants still share one exact integral matrix for all coils.
if response_enabled
    structured = false;
    if common_tau, H = double(ctx.H_response_s(:,:,1)); end
end
if structured
    [~,sample_order] = sort(state_index,'ascend');
else
    sample_order = (1:nsamples).';
end

runtime = ctx;
runtime.base_signal = []; % block caches own the signal; avoid a duplicate
runtime.BT = ctx.B.';
runtime.state_duration_s = [durations 0];
runtime.reference_amplitude = abs(ctx.reference);
runtime.acceleration = struct('Device',device,'BlockSize',block_size, ...
    'TimingStructured',structured,'CachedForward',true,'Precision','double', ...
    'ResponseEnabled',logical(response_enabled),'ResponseCommonTau',logical(common_tau), ...
    'Version',2,'ContributionCacheBytes',16*numel(ctx.base_signal));
template = struct('OriginalSamples',[],'StateIndex',[],'Elapsed_s',[], ...
    'UniqueStates',[],'StateColumns',{{}},'BaseSignal',[],'H_s',[], ...
    'H_response_s',[]);
runtime.blocks = repmat(template,ceil(nsamples/block_size),1);
for block = 1:numel(runtime.blocks)
    first = (block-1)*block_size+1;
    samples = sample_order(first:min(first+block_size-1,nsamples));
    b = template;
    b.OriginalSamples = samples;
    b.StateIndex = state_index(samples).';
    b.Elapsed_s = elapsed(samples).';
    b.UniqueStates = unique(b.StateIndex);
    b.StateColumns = arrayfun(@(state) find(b.StateIndex==state), ...
        b.UniqueStates,'UniformOutput',false);
    b.BaseSignal = ctx.base_signal(:,samples);
    if ~structured, b.H_s = H(samples,:); end
    if response_enabled && ~common_tau
        b.H_response_s = double(ctx.H_response_s(samples,:,:));
    end
    runtime.blocks(block) = b;
end
if strcmp(device,'gpu')
    % gpuDevice is deliberately not reset/selected: preserve the caller's
    % device and let MATLAB report missing/incompatible GPU support.
    gpuDevice;
    runtime.B = gpuArray(double(ctx.B));
    runtime.BT = runtime.B.';
    runtime.state_duration_s = gpuArray(runtime.state_duration_s);
    runtime.reference = gpuArray(ctx.reference);
    runtime.reference_amplitude = gpuArray(runtime.reference_amplitude);
    for block = 1:numel(runtime.blocks)
        runtime.blocks(block).BaseSignal = gpuArray(runtime.blocks(block).BaseSignal);
        runtime.blocks(block).Elapsed_s = gpuArray(runtime.blocks(block).Elapsed_s);
        if ~structured, runtime.blocks(block).H_s = gpuArray(runtime.blocks(block).H_s); end
        if response_enabled && ~common_tau
            runtime.blocks(block).H_response_s = gpuArray(runtime.blocks(block).H_response_s);
        end
    end
end
end
