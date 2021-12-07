% Training of neural network for image reconstruction of digits propagated 
% through multimode fiber

clear all
close all


%% Load training data
% load file "DATA_MMF_16.mat"

data = load("DATA_MMF_16.mat");


% separate data into train, test and val set with a 80-10-10 splitt
x_train  =data.XTrain;
y_train = data.YTrain;


x_test  =data.XTest;
y_test = data.YTest;


x_val  =data.XValid;
y_val = data.YValid;
%% first look at training data
figure
idx = randperm(size(x_train,4),20);
for i = 1:numel(idx)
    subplot(4,5,i)    
    imshow(x_train(:,:,:,idx(i)))
end

%% Create Neural Network Layergraph MLP

I_px = size(x_train,1);
O_px = size(y_train,1);

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
validationFrequency = floor(numel(y_train)/miniBatchSize);



options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{x_val,y_val}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',false);


%%
net = trainNetwork(x_train,y_train,layers,options);
%% Calculate Prediction 
% use command "predict"

%% Evaluate Network
% calculate RMSE, Correlation, SSIM, PSNR

%% Boxplots f�r Aufgabe 6

%% Ab Aufgabe 7: create Neural Network Layergraph U-Net
% Layers = [];

%% Boxplots f�r Aufgabe 8
