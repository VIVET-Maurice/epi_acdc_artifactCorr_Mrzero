function [run_dir, report] = run_experiment_digital_twin(input_file, varargin)
%RUN_EXPERIMENT_DIGITAL_TWIN Compare two acquired EPI images with a Bloch twin.
%   Defaults to Experiment/prepared/experiment_input.mat, exported by the
%   notebook. Creates an isolated output folder; runs no current optimization.
%   Name/value options: OutputDir (new directory only), T2starSeconds ([] uses
%   the exported map; positive scalar including Inf overrides the map).
%   The baseline field is UNKNOWN: B0=0 is an explicit modelling hypothesis.
%   The measured map belongs to the perturbed acquisition and is TOTAL B0,
%   not the added A22 perturbation alone. Only baseline data fit one shared
%   positive gain; the perturbed image is never used to fit the twin/gain.

root = fileparts(mfilename('fullpath'));
if nargin<1 || isempty(input_file)
    input_file = fullfile(root,'prepared','experiment_input.mat');
end
parser = inputParser;
addParameter(parser,'OutputDir','',@(v) ischar(v) || (isstring(v) && isscalar(v)));
addParameter(parser,'T2starSeconds',[],@(v) isempty(v) || ...
    (isnumeric(v) && isreal(v) && isscalar(v) && ~isnan(v) && v>0));
parse(parser,varargin{:});
loaded = load(input_file,'experiment');
assert(isfield(loaded,'experiment'), 'Input MAT must contain the struct experiment.');
experiment = loaded.experiment;
baseline = double(experiment.images.baseline);
perturbed = double(experiment.images.perturbed);
validateattributes(baseline, {'numeric'}, {'real','finite','nonnegative','size',[32 32]});
validateattributes(perturbed, {'numeric'}, {'real','finite','nonnegative','size',[32 32]});
m0 = double(experiment.m0_proxy);
b0 = double(experiment.field.b0Hz);
assert(isequal(size(m0),size(b0),size(experiment.field.mask)), 'Object/map sizes differ.');
valid_field = logical(experiment.field.mask) & isfinite(b0);
assert(any(valid_field(:)), 'There are no reliable B0 voxels.');
simulation_mask = true(size(m0));
if isfield(experiment.field,'simulation_mask')
    assert(isequal(size(experiment.field.simulation_mask),size(m0)), 'Unexpected simulation mask size.');
    simulation_mask = logical(experiment.field.simulation_mask);
end
% Measurement quality and object support are distinct. The notebook supplies
% an explicitly labelled nearest-neighbor B0 extension outside its quality
% mask; removing those object voxels would erase much of the measured object.
m0(~simulation_mask) = 0;
b0(~simulation_mask) = 0;
assert(all(isfinite(b0(:))), 'Export a finite B0 simulation map; preserve measured NaNs separately.');
object_support = simulation_mask & m0>0;
assert(isfield(experiment,'relaxation') && isfield(experiment.relaxation,'T2star_s'), ...
    'Export relaxation.T2star_s, using Inf when no T2* model is available.');
relaxation = experiment.relaxation;
if ~isempty(parser.Results.T2starSeconds)
    relaxation.T2star_s = double(parser.Results.T2starSeconds);
end
if ~isscalar(relaxation.T2star_s)
    assert(isequal(size(relaxation.T2star_s),size(m0)), 'Unexpected T2star map size.');
    relaxation.T2star_s(~simulation_mask) = Inf;
end

[twin_zero, k_zero] = forward_model_experiment(m0,zeros(size(b0)), ...
    experiment.geometry,experiment.acquisition,relaxation);
[twin_measured, k_measured] = forward_model_experiment(m0,b0, ...
    experiment.geometry,experiment.acquisition,relaxation);
predicted = abs(twin_zero);
denominator = sum(predicted(:).^2);
assert(denominator>0, 'The zero-B0 simulation has no signal.');
gain = sum(baseline(:).*predicted(:))/denominator;
assert(isfinite(gain) && gain>0, 'Could not fit a positive baseline-only gain.');
predicted_zero = gain*abs(twin_zero);
predicted_measured = gain*abs(twin_measured);

report = struct;
report.shared_gain = gain;
report.baseline = comparison_metrics(baseline,predicted_zero);
report.perturbed = comparison_metrics(perturbed,predicted_measured);
report.baseline_B0_measured = false;
report.baseline_B0_hypothesis = 'Zero B0; the baseline field was not measured.';
report.perturbed_B0_kind = 'Measured total field during the perturbed GRE acquisition.';
report.receiver_model = 'One uniform receive sensitivity; acquired images use RSS.';
report.m0_kind = 'Effective transverse magnitude at TE, estimated from baseline EPI.';
report.relaxation_model = 'T2star evolution relative to TE; not a separate measured M0/T2 map.';
report.optimization_ready = false;
report.optimization_blocker = 'Experimental ACDC basis maps in Hz/A are not supplied.';
report.reliable_field_voxels = nnz(valid_field & object_support);
report.simulation_voxels = nnz(object_support);
report.measured_field_fraction = report.reliable_field_voxels/max(report.simulation_voxels,1);
report.field_extension = 'Outside the B0 quality mask, notebook nearest-neighbor field extension is an unmeasured modelling assumption.';
report.total_object_voxels = numel(m0);
report.T2star_override_s = parser.Results.T2starSeconds;
report.input_file = char(input_file);
report.acquisition = experiment.acquisition;
if isfield(experiment,'assumptions'), report.export_assumptions = experiment.assumptions; end
if isfield(experiment,'provenance'), report.provenance = experiment.provenance; end

run_dir = char(parser.Results.OutputDir);
if isempty(run_dir)
    results_dir = fullfile(root,'results');
    if ~isfolder(results_dir), mkdir(results_dir); end
    [~,token] = fileparts(tempname(results_dir));
    run_dir = fullfile(results_dir,['digital_twin_' char(datetime('now','Format','yyyyMMdd_HHmmss')) '_' token]);
end
assert(~isfolder(run_dir) && ~isfile(run_dir), 'OutputDir must not already exist.');
[ok,message] = mkdir(run_dir);
assert(ok,'Cannot create output folder: %s',message);
[ok,attributes] = fileattrib(run_dir);
assert(ok,'Cannot resolve output folder.');
run_dir = attributes.Name;
simulation = struct('zero_B0',twin_zero,'measured_B0',twin_measured, ...
    'k_zero_B0',k_zero,'k_measured_B0',k_measured, ...
    'magnitude_zero_B0',predicted_zero,'magnitude_measured_B0',predicted_measured, ...
    'm0_proxy_masked',m0,'b0Hz_masked',b0,'simulation_mask',simulation_mask, ...
    'measured_field_mask',valid_field,'relaxation',relaxation);
save(fullfile(run_dir,'digital_twin_results.mat'),'report','simulation');

fig = figure('Visible','off','Position',[100 100 1000 850]);
closer = onCleanup(@() close(fig)); %#ok<NASGU>
panels = {baseline,perturbed,predicted_zero,predicted_measured};
titles = {'Acquired EPI: no added A22 perturbation', 'Acquired EPI: A22 perturbation', ...
    'Bloch twin: B0=0 hypothesis (not measured)', 'Bloch twin: measured perturbed total B0'};
upper = max([baseline(:);perturbed(:);predicted_zero(:);predicted_measured(:)]);
for panel = 1:4
    subplot(2,2,panel); imagesc(panels{panel}); axis image off;
    title(titles{panel},'Interpreter','none'); clim([0 upper]); colorbar;
end
colormap(fig,'gray');
sgtitle(sprintf('32 x 32 EPI comparison | shared baseline gain %.4g | one receive coil',gain));
savefig(fig,fullfile(run_dir,'digital_twin_comparison.fig'));
exportgraphics(fig,fullfile(run_dir,'digital_twin_comparison.png'),'Resolution',160);
write_report(fullfile(run_dir,'digital_twin_report.txt'),report);
fprintf('Digital twin saved: %s\n',run_dir);
fprintf('Baseline NRMSE %.4f; perturbed NRMSE %.4f; shared gain %.6g.\n', ...
    report.baseline.nrmse,report.perturbed.nrmse,gain);
fprintf('Baseline B0 is unknown; B0=0 is a hypothesis. No ACDC optimization was run.\n');
end

function metrics = comparison_metrics(measured,predicted)
residual = predicted(:)-measured(:);
metrics.rmse = sqrt(mean(residual.^2));
metrics.nrmse = norm(residual)/max(norm(measured(:)),eps);
a = measured(:)-mean(measured(:));
b = predicted(:)-mean(predicted(:));
metrics.correlation = (a.'*b)/max(norm(a)*norm(b),eps);
end

function write_report(path,report)
fid = fopen(path,'w');
assert(fid>=0,'Cannot write report.');
closer = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'Experimental transverse Bloch twin, 32 x 32 EPI\n\n');
fprintf(fid,'Input: %s\nShared positive gain (baseline only): %.9g\n',report.input_file,report.shared_gain);
fprintf(fid,'Baseline NRMSE: %.6g; correlation: %.6g\n',report.baseline.nrmse,report.baseline.correlation);
fprintf(fid,'Perturbed NRMSE: %.6g; correlation: %.6g\n\n',report.perturbed.nrmse,report.perturbed.correlation);
fprintf(fid,'%s\n%s\n%s\n%s\n%s\n',report.baseline_B0_hypothesis, ...
    report.perturbed_B0_kind,report.receiver_model,report.m0_kind,report.relaxation_model);
fprintf(fid,'M0 is zero only outside the exported simulation mask; B0 measurement quality is a separate mask.\n');
fprintf(fid,'Reliable measured B0 covers %d / %d active simulation voxels (%.2f%%).\n', ...
    report.reliable_field_voxels,report.simulation_voxels,100*report.measured_field_fraction);
fprintf(fid,'%s\n',report.field_extension);
fprintf(fid,'RF excitation, TR steady state, slice profile and receive-coil sensitivity are not modelled.\n');
fprintf(fid,'The acquired perturbed EPI is not used to fit M0 or the shared gain.\n');
fprintf(fid,'This is a provisional model comparison; fitting the baseline is not independent validation of the twin.\n');
fprintf(fid,'Optimization ready: false. %s\n',report.optimization_blocker);
if isfield(report,'export_assumptions')
    fprintf(fid,'\nNotebook export assumptions:\n');
    assumptions = report.export_assumptions;
    if ischar(assumptions) || isstring(assumptions), assumptions = cellstr(assumptions); end
    for ii = 1:numel(assumptions)
        fprintf(fid,'- %s\n',char(assumptions{ii}));
    end
end
end
