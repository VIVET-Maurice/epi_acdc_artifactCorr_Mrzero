% Recompute the MATLAB reference used by the comparison notebook.
case_dir = fileparts(mfilename('fullpath'));
addpath(fileparts(case_dir));
load(fullfile(case_dir,'synthetic_matlab_input.mat'));

[img_matlab,kdata_matlab] = forward_model(img,b0,acdc_maps,Ishim,params,seq);
[img_static_matlab,kdata_static_matlab] = forward_model(img,b0,acdc_maps,zeros(size(Ishim)),params,seq);

ix_matlab = round(seq.kx(seq.adc)/seq.dk) + params.matrix_size/2 + 1;
iy_matlab = round(seq.ky(seq.adc)/seq.dk) + params.matrix_size/2 + 1;
valid_matlab = ix_matlab>=1 & ix_matlab<=params.matrix_size & iy_matlab>=1 & iy_matlab<=params.matrix_size;
bins_matlab = sub2ind([params.matrix_size params.matrix_size],ix_matlab(valid_matlab),iy_matlab(valid_matlab));
fprintf('ADC count %d; kept %d; unique bins %d.\n',nnz(seq.adc),nnz(valid_matlab),numel(unique(bins_matlab)));
save(fullfile(case_dir,'synthetic_matlab_output.mat'),'img_matlab','kdata_matlab', ...
    'img_static_matlab','kdata_static_matlab','ix_matlab','iy_matlab','valid_matlab');
