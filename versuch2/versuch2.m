%%% Training of neural network for image reconstruction of digits propagated 
% through multimode fiber

clear all
close all




%% Load training data
% load file "DATA_MMF_16.mat"
%% Load training data
% load file "DATA_MMF_16.mat"
data_old = load("DATA_MMF_16.mat");
%augmeneted dataset :\
data_aug = load("DATA_MMF_16_aug_2.mat");

x_train_old = data_old.XTrain;
y_train_old = data_old.YTrain; 

x_val_old = data_old.XValid;
y_val_old = data_old.YValid;

x_train_aug = data_aug.XTrain;
y_train_aug = data_aug.YTrain;

x_val_aug = data_aug.XValid;
y_val_aug = data_aug.YValid;


x_test = data_old.XTest;
y_test = data_old.YTest;



%%
%datasets = [data_old data_aug];

%for d=1:2
 %   x_train, x_test = datasets(d).XTrain, datasets(d).YTrain ;
    
%end
%% Create Neural Network Layergraph MLP

I_px = size(x_train_old,1);
O_px = size(y_train_old,1);

layers = [imageInputLayer([I_px I_px 1],"Name","Input")
    
fullyConnectedLayer(I_px^2,"Name","Fc1")

reluLayer("Name","Relu1")

fullyConnectedLayer(O_px^2,"Name","Fc2")

reluLayer("Name","Relu2")

depthToSpace2dLayer([O_px O_px],"Name","dts1")

regressionLayer("Name","Ouput")
];




%% Training network
% define "trainingOptions"
% training using "trainNetwork"
miniBatchSize  = 128;
validationFrequency = 5;%floor(numel(y_train)/miniBatchSize);
lr = 0.0001;
%lrdropfactor = 0.000001;
lrdropperiod = 10;
epochs = 10;



%% old data
display('training with old data ..')
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',epochs, ...
    'InitialLearnRate',lr, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{x_val_old,y_val_old}, ...
    'ValidationFrequency',validationFrequency, ...
    'Verbose',false);
[net_old_data, history_old_data] = trainNetwork(x_train_old,y_train_old,layers,options);
save net_old_data;
save history_old_data;

display('saved trainimreading history')
%% aug data
display('training with augmented data ..')
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',epochs, ...
    'InitialLearnRate',lr, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{x_val_aug,y_val_aug}, ...
    'ValidationFrequency',validationFrequency, ...
    'Verbose',false);
[net_aug_data, history_aug_data] = trainNetwork(x_train_aug,y_train_aug,layers,options);
save net_aug_data;
save history_aug_data;

display('saved training history')
%% Calculate Prediction 
% use command "predict"

display('prediciting values ..')
y_test_old = predict(net_old_data, x_test);
y_test_aug = predict(net_aug_data, x_test);
save y_test_old;
save y_test_aug;
display('saved predicted data')
%% Evaluate Network
% calculate RMSE, Correlation, SSIM, PSNR

%% 
display('calculating SSIMs ans PSNRs ..')
ssims_old = zeros(size(y_test,4),1);
ssims_aug = zeros(size(y_test,4),1);

psnr_old = zeros(size(y_test,4),1);
psnr_aug = zeros(size(y_test,4),1);



RMSEs_old =  zeros(size(y_test,4),1);
RMSEs_aug =  zeros(size(y_test,4),1);

for i=1:size(y_test,4)
    imwrite(y_test(:,:,:,i),'ref.jpg');
    imwrite(y_test_old(:,:,:,i),'predold.jpg');
    imwrite(y_test_aug(:,:,:,i),'predaug.jpg');
    pic_old = imread('predold.jpg');
    pic_ref = imread('ref.jpg');
    pic_aug = imread('predaug.jpg');
    
    ssims_old(i) = ssim(pic_old,pic_ref);
    ssims_aug(i) = ssim(pic_aug,pic_ref);
    
    psnr_old(i) = psnr(pic_old, pic_ref);
    psnr_aug(i) = psnr(pic_aug, pic_ref);
    
end

save ssims_old;
save ssims_aug;
save psnr_old;
save psnr_aug;

display('saved SSIMs')
%% correlation ceoff

display('calculating corellation coeff ..')


corr_old = zeros(size(y_test,4),1);
corr_aug = zeros(size(y_test,4),1);



RMSEs_old =  zeros(size(y_test,4),1);
RMSEs_aug =  zeros(size(y_test,4),1);


for i=1:size(y_test,4)

    y_old = y_test_old(:,:,1,i);
    y_aug = y_test_aug(:,:,1,i);
    y_t = y_test(:,:,1,i);
    coef_old = corrcoef(y_old(:),y_t(:));
    coef_old = coef_old(1,2);
    coef_aug = corrcoef(y_aug(:),y_t(:));
    coef_aug = coef_aug(1,2);
    corr_old(i) = coef_old;
    corr_aug(i) = coef_aug;
    
    
    
    RMSE_old =  sqrt(mean((y_old-y_t).^2));
    RMSE_aug = sqrt(mean((y_aug-y_t).^2));

    RMSEs_old(i) = RMSE_old
    RMSES_aug(i) = RMSE_aug 
   
    
end


save corr_old;
save corr_aug;
save RMSEs_old;
save RMSEs_aug;

display('saved corellation coeff')
%% Boxplots f�r Aufgabe 6
 ssims = [ssims_old, ssims_aug];
 corrs = [corr_old, corr_aug];
 psnrs =[psnr_old, psnr_aug];
 
 boxplot(ssims); ylabel('SSIM');
 legend('old data', 'augmented data');
 saveas (gcf,'boxplotSSIM.jpg')
 
 
 boxplot(corrs); ylabel('CORR');
 legend('old data', 'augmented data');
 saveas (gcf,'boxplotCORR.jpg')
 

 
 
 boxplot(psnrs); ylabel('PSNR');
 legend('old data', 'augmented data');c
 saveas (gcf,'boxplotPSNR.jpg')
 
%% Ab Aufgabe 7: create Neural Network Layergraph U-Net
% Layers = [];






unet_layers = unetLayers([I_px I_px 1],2,...
"encoderDepth",3);

finalConvLayer = convolution2dLayer(1,1,"Padding","same","Stride",1,"Name","Final-ConvolutionLayer");


unet_layers = replaceLayer(unet_layers,...
"Final-ConvolutionLayer",finalConvLayer);

unet_layers = removeLayers(unet_layers,"Softmax-Layer");

regLayer = regressionLayer("Name","Reg-Layer");

unet_layers = replaceLayer(unet_layers,...
"Segmentation-Layer",regLayer);

unet_layers = connectLayers(unet_layers,...
"Final-ConvolutionLayer","Reg-Layer");

display('train unet ..')

[unet, history_unet] =  trainNetwork(x_train_aug,y_train_aug,unet_layers,options);

save unet;
save history_unet;


display('finished unet training')
%%


display('calculating SSIMs ans PSNRs ..')
y_unet = predict(unet, x_test);
ssims_u = zeros(size(y_test,4),1);
corr_u = zeros(size(y_test,4),1);
psnr_u = zeros(size(y_test,4),1);
RMSEs_u = zeros(size(y_test,4),1);
for i=1:size(y_test,4)
    imwrite(y_test(:,:,:,i),'ref.jpg');
    imwrite(y_unet(:,:,:,i),'unet.jpg');
    
    pic_u = imread('unet.jpg');
    pic_ref = imread('ref.jpg');
   
    ssims_u(i) = ssim(pic_u,pic_ref);
    
    psnr_u(i) = psnr(pic_u, pic_ref);
    y_u = y_unet(:,:,1,i);
    

    y_t = y_test(:,:,1,i);
    coef_u = corrcoef(y_u(:),y_t(:));
    
    corr_u(i) = coef_u(1,2);
    
    RMSEs_u =  sqrt(mean((y_u-y_t).^2));
    
end

save y_unet;
save corr_u;
save psnr_u;
save ssims_u;
save RMSEs_u;
    

%% Boxplots f�r Aufgabe 8

% ssimss boxplot

boxplot([RMSE_aug RMSEs_u]); ylabel('RMSE');
legend('MLP','unet');
saveas (gcf,'boxplotUnetRMSE.jpg');



boxplot([ssims_aug ssims_u]); ylabel('SSIM');
legend('MLP','unet');
saveas (gcf,'boxplotUnetSSIM.jpg');


boxplot([psnr_aug psnr_u]); ylabel('PSNR');

legend('MLP','unet');
saveas (gcf,'boxplotUnetPSNR.jpg');

boxplot([corr_aug corr_u]); ylabel('CORR');

legend('MLP','unet');
saveas (gcf,'boxplotUnetCORR.jpg');
