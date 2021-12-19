%%% Training of neural network for image reconstruction of digits propagated 
% through multimode fiber

clear all
close all




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

%% aug data
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


%% Calculate Prediction 
% use command "predict"
y_test_old = predict(net_old_data, x_test);
y_test_aug = predict(net_aug_data, x_test);
%% Evaluate Network
% calculate RMSE, Correlation, SSIM, PSNR
RMSE_old = sqrt(mean((y_test_old-y_test).^2));
RMSE_aug = sqrt(mean((y_test_aug-y_test).^2));
boxplot(RMSE_old(4))
hold on
boxplot(RMSE_aug(4))
% [ssim_old,a] = ssim(y_test_old, y_test);


ref = imshow(y_test(:,:,:,1));
A = imshow(y_test_old(:,:,:,1));
ssim = ssim(A,ref);


%% 
ssims_old = [];
ssims_aug = [];
for i=1:size(y_test,4)
    imwrite(y_test(:,:,:,i),'ref.jpg');
    imwrite(y_test_old(:,:,:,i),'predold.jpg');
    imwrite(y_test_aug(:,:,:,i),'predaug.jpg');

    ssims_old(i) = ssim(imread('predold.jpg'),imread('ref.jpg'));
    ssims_aug(i) = ssim(imread('predaug.jpg'),imread('ref.jpg'));
end

%%
xcorr_old = xcorr(y_test_old);
xcorr_aug = xcorr(y_test_aug);
xcorr_test = xcorr(y_test)
%% Boxplots f�r Aufgabe 6

%% Ab Aufgabe 7: create Neural Network Layergraph U-Net
% Layers = [];

%% Boxplots f�r Aufgabe 8
