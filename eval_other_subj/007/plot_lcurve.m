

clear;

load sol_BASELINE.mat;

field_limit = [ 25 100 200 300 500 1000 ]';

imgs = cell( numel(field_limit) , 1 );
normFac = mean(abs(img_ref(:)).^2);
y = [];

for ii = 1 : numel(field_limit)
    
    load( sprintf( 'sol_shimOpt3_ACDCFieldLimit%d.mat' , field_limit(ii) ) ); 
    imgs{ii,1} = abs(img_acdc2);
    close all;
      
    err = imgs{ii,1} - abs(img_ref);

    y = [ y  ;  sqrt(mean(err(:).^2) / normFac ) * 100  ] ;
    
end


figure;  hold on;
err2sh = abs(img_2sh) - abs(img_ref);
erracdc = abs(img_acdc) - abs(img_ref);

err2sh = sqrt( mean(err2sh(:).^2) / normFac ) * 100;
erracdc = sqrt( mean(erracdc(:).^2) / normFac ) * 100;


x = field_limit;

plot( [min(x)  max(x)] , [ err2sh  err2sh ] , '--b' , 'lineWidth' , 2 );
plot( [min(x)  max(x)] , [ erracdc  erracdc ] , '--k' , 'lineWidth' , 2 );

plot( x, y , '-or'  , 'lineWidth' , 2 );
xlabel( 'max ACDC field [Hz]' );
ylabel( 'RMSE [%]' );
box off;

legend( '2SH' , 'ACDC' , 'ACDC field opt.' , 'Location' , 'best' );


figure;
imagesc( abs([img_ref  img_2sh  img_acdc  imgs{3}  imgs{end}]) );
axis image off;
title( 'Ref/2SH/ACDC/ACDC field opt. (200 Hz)/ACDC field opt. (1000 Hz)' );














