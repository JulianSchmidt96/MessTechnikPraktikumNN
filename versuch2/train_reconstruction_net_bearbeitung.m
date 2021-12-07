% Training of neural network for image reconstruction of digits propagated 
% through multimode fiber

clear all
close all




%% Load training data
% load file "DATA_MMF_16.mat"

%data = load("DATA_MMF_16.mat");
%augmeneted dataset :\
data = load("DATA_MMF_16_aug_2.mat");


x_train  =data.XTrain;
y_train = data.YTrain;


x_test  =data.XTest;
y_test = data.YTest;


x_val  =data.XValid;
y_val = data.YValid;


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
validationFrequency = 5;%floor(numel(y_train)/miniBatchSize);
lr = 0.0001;
%lrdropfactor = 0.000001;
lrdropperiod = 10;
epochs = 200;


%%


options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',epochs, ...
    'InitialLearnRate',lr, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{x_val,y_val}, ...
    'ValidationFrequency',validationFrequency, ...
    'Verbose',false);


%%
[net, history] = trainNetwork(x_train,y_train,layers,options);
save net;
save history;
%%
valloss = history.ValidationLoss;
valloss(find(isnan(valloss)))=[];

trainloss = history.TrainingLoss;
trainloss(find(isnan(trainloss)))=[];

%%
plot(valloss(5:end))

%% Calculate Prediction 
% use command "predict"

%% Evaluate Network
% calculate RMSE, Correlation, SSIM, PSNR

%% Boxplots f�r Aufgabe 6

%% Ab Aufgabe 7: create Neural Network Layergraph U-Net
% Layers = [];

%% Boxplots f�r Aufgabe 8