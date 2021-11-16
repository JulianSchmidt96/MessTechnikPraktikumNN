%% Load Training Data & define class catalog & define input image size
disp('Loading training data...')
% download from MNIST-home page or import dataset from MATLAB
% https://www.mathworks.com/help/deeplearning/ug/data-sets-for-deep-learning.html
% http://yann.lecun.com/exdb/mnist/

% Specify training and validation data
% Recommended naming >>>
% Train: dataset for training a neural network
% Test: dataset for test a trained neural network after training process
% Valid: dataset for test a trained neural network during training process
% X: input / for Classification: image
% Y: output / for Classification: label
% for example: XTrain, YTrain, XTest, YTest, XValid, YValid

%httpsUrl = 'http://yann.lecun.com/exdb/mnist';
%imageUrl = strcat(httpsUrl, '/train-images-idx3-ubyte.gz');
%XTrain = webread(imageUrl);

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
fullDataset = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% XTrain = fullDataset.Files{:};   %Image
% YTrain = fullDataset.Labels{:};   %Label
% XTest = fullDataset.Files{:};    %Image
% YTest = fullDataset.Labels{:};    %Label
% XValid =fullDataset.Files{:};    %Image
% YValid = fullDataset.Labels{:};    %Label

figure;
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end
labelCount = countEachLabel(imds)

img = readimage(imds,1);
size(img)

numTrainFiles = 750;
[TrainSet,ValidationSet] = splitEachLabel(fullDataset,numTrainFiles,'randomize');
%% define network (dlnet)
NN_layers = [
    imageInputLayer([28 28 1],'Normalization','none','Name','input')
    
    fullyConnectedLayer(1000, 'Name','fullyConnected1')
    reluLayer('Name','relu1')
    
    fullyConnectedLayer(10,'Name','fullyConnected2')
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','classification')
   ];

% % convert to a layer graph
% lgraph = layerGraph(NN_layers);
% % Create a dlnetwork object from the layer graph.
% 
% dlnet = dlnetwork(lgraph);
% % visualize the neural network
% analyzeNetwork(dlnet)

%% Specify Training Options (define hyperparameters)

% miniBatchSize
% numEpochs
% learnRate 
% executionEnvironment
% numIterationsPerEpoch 

% training on CPU or GPU(if available);
% 'auto': Use a GPU if one is available. Otherwise, use the CPU.
% 'cpu' : Use the CPU
% 'gpu' : Use the GPU.
% 'multi-gpu' :Use multiple GPUs
% 'parallel :

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',ValidationSet, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(TrainSet,NN_layers,options);

[net, lr_001] =  trainNetwork(TrainSet,NN_layers,options);

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',ValidationSet, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(TrainSet,NN_layers,options);

[net, lr_0001] =  trainNetwork(TrainSet,NN_layers,options);

%% 
disp(lr_001)
disp(lr_0001)
x = reshape(lr_001.ValidationLoss(:, 1:289), [], 1)
y = reshape(lr_0001.ValidationLoss(:, 1:289), [], 1)

x(find(isnan(x)))=[]
y(find(isnan(y)))=[]

plot(x)
hold on
plot(y)

legend('lr 0,001','lr 0,0001')
%% Train neural network

% initialize the average gradients and squared average gradients
% averageGrad
% averageSqGrad

% "for-loop " for training

for epoch = 1:numEpochs
    
   % updae learnable parameters based on mini-batch of data
    for i = 1:numIterationsPerEpoch
        % Read mini-batch of data and convert the labels to dummy variables.


        % Convert mini-batch of data to a dlarray.
        
        
        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients helper function.
        
        % Update the network parameters using the optimizer, like SGD, Adam
        
        % Calculate accuracy & show the training progress.

        % option: validation

    end
end


%% test neural network & visualization 

%% Define Model Gradients Function
% 
function [gradients,loss,dlYPred] = modelGradients(dlnet,dlX,Y)

    % forward propagation 
    dlYPred = forward(dlnet,dlX);
    % calculate loss -- varies based on different requirement
    loss = crossentropy(dlYPred,Y);
    % calculate gradients 
    gradients = dlgradient(loss,dlnet.Learnables);
    
end
