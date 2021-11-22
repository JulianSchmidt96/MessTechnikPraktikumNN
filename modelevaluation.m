%% Fetch Dataset
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
%%

opt = 'adam';

ac = zeros(size(lrs));
for lr = 1:length(lrs)
    modelOptions = trainingOptions('opt', ...
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',ValidationSet, ...
    'ValidationFrequency',30, ...
    'Verbose',false);
    [net, results] = trainNetwork(TrainSet,NN_layers,options);

    
    x = results.FinalValidationAccuracy;
    ac(:,lr) = x;
    
    
end
%% Get best result
[val,in] = max(ac);
sprintf('%.15g  is the best accuracy and was achieved with the learning rate %.15g and the optimizer %s',val,lrs(:,in), opt)
best_lr = lrs(:,in);
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
