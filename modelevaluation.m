tic
   

%% Fetch Dataset 75/25 train/val splitt
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
fullDataset = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
numTrainFiles = 750;
[TrainSet,ValidationSet] = splitEachLabel(fullDataset,numTrainFiles,'randomize');

%% def Model



NN_layers = [
    imageInputLayer([28 28 1],'Normalization','none','Name','input')
    
    fullyConnectedLayer(1000, 'Name','fullyConnected1')
    reluLayer('Name','relu1')
    
    fullyConnectedLayer(10,'Name','fullyConnected2')
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','classification')
   ];


%%
start_lr = 10^-6;
end_lr = 10^-1;
stepmulti = 10;
lrs = fetchlearningrates(start_lr, end_lr, stepmulti);
epochs = 5;
%%

opts = {'adam','sgdm'};
for opt =1:length(opts)

    ac = zeros(size(lrs));
    for lr = 1:length(lrs)
        modelOptions = trainingOptions(opts{:,opt}, ...
        'InitialLearnRate',lr, ...
        'MaxEpochs',epochs, ... 
        'ValidationData',ValidationSet, ...
        'ValidationFrequency',30, ...
        'Verbose',false);
        [net, results] = trainNetwork(TrainSet,NN_layers,options);
    
        
        x = results.FinalValidationAccuracy;
        ac(:,lr) = x;
        
        
    end

[val,in] = max(ac);
sprintf('%.15g  is the best accuracy and was achieved with the learning rate %.15g for the optimizer %s',val,lrs(:,in), opts{opt})
best_lr = lrs(:,in);
end
toc
%% 
function lrs = fetchlearningrates(start_lr, end_lr, stepmulti)
    i=start_lr;
    k = 1;

    lrs = [];

    while i~=end_lr
       lrs(k) = i;
       i = i * stepmulti;
       k = k + 1;
    end
     
end
