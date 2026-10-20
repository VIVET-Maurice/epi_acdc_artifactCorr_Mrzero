

function f = obj_fun( x , img , B0Hz , shimHzPerAmp , img_ref , params , seq , mask , shim_opt )

Nadc = params.matrix_size;
Nc   = size(shimHzPerAmp,3);

if shim_opt == 1
    Ishim = repmat(x,[1 Nadc]);
elseif shim_opt == 2
    Ishim = zeros(Nadc,Nc);
    Ishim(:,1:2:end) = repmat( x(1:Nc)      , [ 1 Nadc/2 ] );
    Ishim(:,2:2:end) = repmat( x(Nc+1:end)  , [ 1 Nadc/2 ] );
else
    Ishim = reshape(x,[Nc Nadc]);
end

img_hat = forward_model( img , B0Hz , shimHzPerAmp , Ishim , params , seq );

img_hat = abs(img_hat);
img_ref = abs(img_ref);
% img_hat = img_hat / mean(img_hat(find(mask))) * mean(img_ref(find(mask)));

err = img_hat - img_ref;

f = mean(err(find(mask)).^2);

end