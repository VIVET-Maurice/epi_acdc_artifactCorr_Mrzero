


% clear;

addpath( '/autofs/space/guerin/CODE/iso2mesh/' );
addpath( '/autofs/space/guerin2/public/GUERIN/TOY_MODELS/epi_acdc_artifactCorr/' )


% load solutions for subject 006
field_limit = [ 25 100 200 300 500 1000 ]';
Ishim_006 = cell( numel(field_limit) , 1 );
for ii = 1 : numel(field_limit)
    load( sprintf( '../../sol_shimOpt3_ACDCFieldLimit%d.mat' , field_limit(ii) ) );
    close all;
    Ishim_006{ii , 1} = Ishim2;
end

subj = 007;

load( sprintf( '/autofs/space/guerin/theodore/b0_shimming/shimopt/for_people/for_bastien/b0_maps_acdc/b0maps/b0db_%03d.mat' , subj ) );
% load /autofs/space/guerin/theodore/b0_shimming/shimopt/for_people/for_bastien/b0_maps_acdc/acdc32_basis;

slice = 120;
mask  = double( field_map_data.mask(:,:,slice) );
[ ix  iy ] = ind2sub( size(mask) , find(mask) );
x1 = min(ix);  x2 = max(ix);
y1 = min(iy);  y2 = max(iy);
usfac = 3;
mask  = double( field_map_data.mask(x1:usfac:x2,y1:usfac:y2,slice) );
bmask = double( field_map_data.brain_mask(x1:usfac:x2,y1:usfac:y2,slice) );
b0    = field_map_data.field_map_ppm(x1:usfac:x2,y1:usfac:y2,slice) * 297 .* mask;  % field map [Hz]
acdc_maps = squeeze( acdc_basis(x1:usfac:x2,y1:usfac:y2,slice,:) ) * 42.6e6;   % .* mask; 
params.X  = field_map_data.x_pos_mm(x1:usfac:x2,y1:usfac:y2,slice) * 0.001;
params.Y  = ( field_map_data.y_pos_mm(x1:usfac:x2,y1:usfac:y2,slice) - 10 ) * 0.001;

% % input image is resolution pattern with mask contour
% resPat = 5;
% img  = zeros(size(mask)); img(1:resPat:end,:) = 1;  img(:,1:resPat:end) = 1;   % resolution pattern
% fmask = imfill(mask);
% img2 = fmask - thinbinvol(fmask,1); 
% img = double( ( img + img2 ) > 0 ) .* mask;

% input image is MPRAGE
img = field_map_data.mprage(x1:usfac:x2,y1:usfac:y2,slice);


params.FOV = 0.22;        % EPI FOV [m]
params.matrix_size = 32; % EPI matrix size
params.dt = 8e-6;        % ADC sampling time [s]
params.Gmax = 0.08;      % T/m
params.Smax = 200;       % T/m/s
params.T1 = 1.2;         % s
params.T2 = 0.080;       % s
params.Imax = 5.0;       % maximum ACDC current per channel [A]

figure; 
subplot(2,2,1); imagesc(img); axis image off; title('image'); colorbar;
subplot(2,2,2); imagesc(b0); axis image off; title('B0 [Hz]'); colorbar; caxis([-200 200]);
subplot(2,2,3); imagesc(mask); axis image off; title('mask'); colorbar; 

tp1 = sum(abs(acdc_maps),3); tp1 = tp1 / max(tp1(find(mask)));
tp2 = mask;
subplot(2,2,4); imagesc(tp1 + tp2); axis image off; title('ACDC maps + image'); caxis([0 2]);  colorbar


% CREATE EPI WAVEFORMS
seq = make_cartesian_epi_sequence( params , 1 );


% ACDC DYNAMIC SHIMMING
[ I0  results ] = initialize_acdc_currents( b0 , acdc_maps , -params.Imax*0.99 , params.Imax*0.99 , mask , 1 );
Ishim = repmat( I0 , [ 1 params.matrix_size ] );
img_ref  = forward_model( img , zeros(size(b0)) , acdc_maps , zeros(size(Ishim)) , params , seq );
img_2sh  = forward_model( img , b0              , acdc_maps , zeros(size(Ishim)) , params , seq );
img_acdc = forward_model( img , b0              , acdc_maps , Ishim              , params , seq );

figure; 
subplot(2,2,1);  imagesc(img); axis image off; title('Input image');
subplot(2,2,2);  imagesc(abs(img_ref)); axis image off; title('EPI ref (B0=0)');
subplot(2,2,3);  imagesc(abs(img_2sh)); axis image off; title('EPI 2SH');
subplot(2,2,4);  imagesc(abs(img_acdc)); axis image off; title('EPI 2SH + ACDC shimming');
pause(0.1);

save  sol_BASELINE.mat  Ishim  img_ref  img_2sh  img_acdc  params  seq;


% RUN OPTIMIZATION
% shim_opt controls the shim strategy
% 1: constant shim during sequence
% 2: alternate shim between EPI lobes (two shim states for even and odd echoes)
% 3: one shim state per adc event
params.shim_opt = 3;
params.niters = 500;

% downsample head mask to reduce number of linear constraints
mask2 = zeros(size(mask));
mask2( 1:3:end , 1:3:end ) = mask( 1:3:end , 1:3:end );

for ii = 1 : numel(field_limit)

    fprintf( '   => ACDC field limit = %d\n' , field_limit(ii) );

    params.max_acdc_field = field_limit(ii);  % [Hz]
    
    % [ Ishim2  Ishim2_fields ] = epi_acdc_artCor( img , b0 , acdc_maps , img_ref , I0 , params , seq , mask2 );

    % plot and save results
    img_acdc2 = forward_model( img , b0 , acdc_maps , Ishim_006{ii} , params , seq );
    
    F1 = figure; 
    subplot(2,1,1);  plot(Ishim_006{ii});
    box off; 
    xlabel('ADC index');
    ylabel('ACDC current [A]');
    subplot(2,2,3);  imagesc(abs(img_acdc)); axis image off; title('Conventional ACDC shimming');
    subplot(2,2,4);  imagesc(abs(img_acdc2)); axis image off; title('Image-based ACDC shimming');
    
    F2 = figure; imagesc(results.shimFieldHz); axis image off; colorbar; caxis([-200 200]);
    title('Conventional ACDC shim field');
    
    cmd = sprintf( 'save sol_shimOpt%d_ACDCFieldLimit%d.mat  Ishim2  img_acdc2  params  seq  F1  F2;' , ...
        params.shim_opt , field_limit(ii));
    eval(cmd);

end
