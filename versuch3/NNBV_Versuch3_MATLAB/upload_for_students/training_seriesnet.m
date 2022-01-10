% Training of seriesnet-type neural network for mode decomposition
clear all

%% load dataset
%  1. load the dataset
load('/home/schmijul/MessTechnikPraktikumNN/MessTechnikPraktikumNN/versuch3/NNBV_Versuch3_MATLAB/upload_for_students/trainset.mat');
%  2. define the input and output size for neural network
x_train = trainset.images;
y_train = trainset.labels;
x_size = size(x_train, 1, 2);
y_size = size(y_train, 1, 2);

%% create MLP neural network - Aufgabe 3
% Layers_MLP = [];

%% create VGG neural network - Aufgabe 6
% Layers_VGG= [];

%% Training network
% define "trainingOptions"
% training using "trainNetwork"

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

