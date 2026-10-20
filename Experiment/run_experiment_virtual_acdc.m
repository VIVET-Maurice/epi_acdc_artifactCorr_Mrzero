function [run_dir,summary] = run_experiment_virtual_acdc(device,varargin)
%RUN_EXPERIMENT_VIRTUAL_ACDC Real phantom B0 with a simulated 3T ACDC coil.
%   run_experiment_virtual_acdc('gpu','MaxIterations',3,'FieldLimits',100)
%   CPU: up to five independent limits, six threads each (31 total <32).
%   GPU: one worker with six CPU threads; one shared GPU cache, double.
%   Coil placement is the documented identity transform in patient space.
%   The separate input retains only points covered by the measured maps.
%   No ACDC coil was present in the phantom scan; results are virtual.
if nargin<1 || isempty(device), device = 'cpu'; end
device = validatestring(lower(char(device)),{'cpu','gpu'});
root = fileparts(mfilename('fullpath'));
prepared = fullfile(root,'prepared','acdc_calibration');
if strcmp(device,'gpu'), workers = 1; else, workers = 5; end
[run_dir,summary] = run_experiment_fast_parallel( ...
    fullfile(prepared,'acdc_calibration_candidate.mat'),1,workers, ...
    'InputFile',fullfile(prepared,'experiment_input_acdc_support.mat'), ...
    'TotalCurrentLimit_A',30,'CalibrationMode','virtual', ...
    'ThreadsPerWorker',6,'Device',device,'MakeFigures',true,varargin{:});
end
