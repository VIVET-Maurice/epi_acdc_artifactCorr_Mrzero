



function [ Isol  Isol_fields ] = epi_acdc_artCorr( img , b0 , acdc_maps , img_ref , I0 , params , seq , head_mask )

Nadc = params.matrix_size;
Nc   = size(acdc_maps,3);

x0 = I0;
if params.shim_opt == 2
    x0 = [ x0 ; x0 ];
elseif params.shim_opt == 3
    x0 = repmat( x0 , [Nadc 1] );  % fast index: channels, slow index: ADC
end

opt_mask = ones(size(img_ref));
objfun = @(x) obj_fun( x , img , b0 , acdc_maps , img_ref , params , seq , opt_mask , params.shim_opt );

options = optimoptions('fmincon', ...
    'Algorithm','interior-point', ...
    'SpecifyObjectiveGradient',false, ...
    'Display','iter-detailed', ...
    'MaxIterations',params.niters, ...
    'MaxFunctionEvaluations',1e10, ...
    'OptimalityTolerance',1e-6, ...
    'StepTolerance',1e-6);

fprintf( '**** START OPTIMIZATION F(x0)=%e ****\n' , objfun(x0) );

Imin = -params.Imax * ones(size(x0));
Imax =  params.Imax * ones(size(x0));

% constrain maximum ACDC correction inside the head mask
A = permute( acdc_maps , [3 1 2] );
A = A( : , find(head_mask) );
A = permute( A , [2 1] );

if params.shim_opt == 2

elseif params.shim_opt == 3
    blocks = cell(Nadc,1);
    for nn = 1 : Nadc
        blocks{nn} = sparse(A);
    end
    A = blkdiag(blocks{:});
end

A2 = [ A ; -A ];
b2 = ones(size(A2,1),1) * params.max_acdc_field;

[xopt,fval,exitflag,output] = fmincon( objfun , x0 , A2 , b2 , [] , [] , Imin , Imax , [] , options );

if params.shim_opt == 1
    Isol = repmat(xopt,[1 Nadc]);
elseif params.shim_opt == 2
    Isol = zeros(Nc,Nadc);
    Isol(:,1:2:end) = repmat( xopt(1:Nc)     , [ 1  Nadc/2 ] );
    Isol(:,2:2:end) = repmat( xopt(Nc+1:end) , [ 1  Nadc/2 ] );
else
    Isol = reshape(xopt,[Nc Nadc]);
end

[ Nx  Ny ] = size(img);
Isol_fields  = zeros(Nx,Ny);
if params.shim_opt == 1
    Isol_fields  = zeros(Nx,Ny);
elseif params.shim_opt == 2
    Isol_fields  = zeros(Nx,Ny,2);
else
    Isol_fields  = zeros(Nx,Ny,Nadc);
end

for ii = 1 : Nc
    if params.shim_opt == 1
        Isol_fields = Isol_fields + acdc_maps(:,:,ii) * Isol(ii,1);
    elseif params.shim_opt == 2
        Isol_fields(:,:,1) = Isol_fields(:,:,1) + acdc_maps(:,:,ii) * Isol(ii,1);
        Isol_fields(:,:,2) = Isol_fields(:,:,2) + acdc_maps(:,:,ii) * Isol(ii,2);
    else
        for jj = 1 : Nadc
            Isol_fields(:,:,jj) = Isol_fields(:,:,jj) + acdc_maps(:,:,ii) * Isol(ii,jj);
        end
    end

end






