
%% Fetch Dataset 75/25 train/val splitt


digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
fullDataset = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
numTrainFiles = 750;
[TrainSet,ValidationSet] = splitEachLabel(fullDataset,numTrainFiles,'randomize');

%% def Model


NN_layer = [
    imageInputLayer([28 28 1],'Normalization','none','Name','input')
    
    fullyConnectedLayer(1000, 'Name','fullyConnected1')
    reluLayer('Name','relu1')
    
    fullyConnectedLayer(10,'Name','fullyConnected2')
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','classification')
   ];


%% Hyperparams


start_mbatch = floor(255/6);
end_mbatch = floor(255/6) * 6;

mbatches = addedarray(start_mbatch, end_mbatch, start_mbatch);
start_lr = 10^-6;
end_lr = 10^-1;

lrs = multiplicatedarray(start_lr, end_lr, 10);
epochs = 20;

val_freq = 10;

%% eval


opts = {'adam','sgdm'};
optimizier = opts{1};
for opt =1:length(opts)

    ac = zeros(size(lrs));
    for lr = 1:length(lrs)
        modelOptions = trainingOptions(opts{:,opt}, ...
        'InitialLearnRate',lr, ...
        'MaxEpochs',epochs, ... 
        'Shuffle','every-epoch', ...
        'ValidationData',ValidationSet, ...
        'ValidationFrequency',val_freq, ...
        'Verbose',false);
        [net, results] = trainNetwork(TrainSet,NN_layer,modelOptions);
    
        
        x = results.FinalValidationAccuracy;
        ac(lr) = x;
        
        
    end

valacc = 0;

if max(ac) > valacc
    optimizier = opts(opt);
    [valacc,inlr] = max(ac);
end
end
sprintf('%.15g  is the best accuracy and was achieved with the learning rate %.15g and the optimizer %s',valacc,lrs(:,inlr), opts{opt})
best_lr = lrs(:,inlr);



% working with different mini batch sizes

mean_val = zeros(size(mbatches));
times = zeros(size(mbatches));
for mb = 1:length(mbatches)
       
modelOptions = trainingOptions(opts{:,opt}, ...
        'InitialLearnRate',lr, ...
        'MaxEpochs',epochs, ... 
        'ValidationData',ValidationSet, ...
        'Shuffle','every-epoch', ...
        'ValidationFrequency',val_freq, ...
        'MiniBatchSize',mbatches(mb), ...
        'Verbose',false);
        %[net, results] = trainNetwork(TrainSet,NN_layer,modelOptions);
        f = @() trainNetwork(TrainSet,NN_layer,modelOptions);
        time = timeit(f);
        %x = results.ValidationAccuracy;
       % x(find(isnan(x)))=[];
        %mean_val(mb) = mean(x);
        times(mb) = time;
    
end
%[val,in] = max(mean_val);
[valtime,inmb] =min(times);
sprintf('%.15g seconds is the fastest training time with a minibatchsize of  minibatchsize = %.15g with the learning rate %.15g and the optimizer %s',valtime,mbatches(inmb), best_lr, opts{opt})




%% train again with best hyperparams

epochs = 30;

modelOptions = trainingOptions(opts{:,opt}, ...
        'InitialLearnRate',lrs(inlr), ...
        'MaxEpochs',epochs, ... 
        'ValidationData',ValidationSet, ...
        'Shuffle','every-epoch', ...
        'ValidationFrequency',val_freq, ...
        'MiniBatchSize',mbatches(inmb), ...
        'Verbose',false, ...
        'Plots','training-progress');
    
[net, results] = trainNetwork(TrainSet,NN_layer,modelOptions);
save(net)
save(results)
%% evaluate accuracy
YPred = classify(net,ValidationSet);
YValidation = ValidationSet.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
%% 

function arr = addedarray(startval, endval, stepadd)
    i=startval;
    k = 1;
    
    arr = [];

    while i~=endval
        arr(k) = i;
       i = i + stepadd;
       k = k + 1;
    end
     
end
function arr = multiplicatedarray(startval, endval, stepmulti)
    i=startval;
    k = 1;

    arr = [];

    while i~=endval
       arr(k) = i;
       i = i * stepmulti;
       k = k + 1;
    end
     
end
