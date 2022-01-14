% Training of seriesnet-type neural network for mode decomposition
clear all

%% load dataset
%  1. load the dataset
load('/home/schmijul/MessTechnikPraktikumNN/MessTechnikPraktikumNN/versuch3/NNBV_Versuch3_MATLAB/upload_for_students/trainset.mat');

%  2. define the input and output size for neural network

x_train = abs(trainset.images);
y_train = trainset.labels;
x_size = size(x_train, 1, 2, 3);
y_size = size(y_train, 1, 2);
input_size = x_size(1,2);%,x_size(1);
output_size = 5;%;size(y_train(:,1)) ;

 
%% create MLP neural network - Aufgabe 3
 
Layers_MLP=[imageInputLayer([x_size],"Name","Input")
fullyConnectedLayer(input_size*input_size, 'Name', 'fc1')
leakyReluLayer('Name', 'relu1')
fullyConnectedLayer(input_size*input_size,'Name','fc2')
leakyReluLayer('Name', 'relu2')
fullyConnectedLayer(output_size,"Name","fc_output")
regressionLayer("Name","Ouput")];


%% create VGG neural network - Aufgabe 6
% Layers_VGG= [];

%% Training network
% define "trainingOptions"

options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',64, ...
    'Plots','training-progress');


% training using "trainNetwork"
[mlp, mlp_history] = trainNetwork(x_train,y_train,Layers_MLP,options);



%% Test Network  - Aufgabe 4
% use command "predict"

% reconstruct field distribution
% [] = mmf_rebuilt_image();

%%  Visualization results  - Aufgabe 5
% calculate Correlation between the ground truth and reconstruction
% calculate std
% plot()
% calulate relative error of ampplitude and phase 


%% save model

