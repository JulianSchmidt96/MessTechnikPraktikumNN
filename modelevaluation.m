
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


%% Hyperparams


start_mbatch = floor(255/6);
end_mbatch = floor(255/6) * 6;

mbatches = addedarray(start_mbatch, end_mbatch, start_mbatch);
start_lr = 10^-6;
end_lr = 10^-1;

lrs = multiplicatedarray(start_lr, end_lr, 10);
epochs = 5;

val_freq = 30;

%% eval


opts = {'adam','sgdm'};
for opt =1:length(opts)

    ac = zeros(size(lrs));
    for lr = 1:length(lrs)
        modelOptions = trainingOptions(opts{:,opt}, ...
        'InitialLearnRate',lr, ...
        'MaxEpochs',epochs, ... 
        'ValidationData',ValidationSet, ...
        'ValidationFrequency',val_freq, ...
        'Verbose',false);
        [net, results] = trainNetwork(TrainSet,NN_layers,modelOptions);
    
        
        x = results.FinalValidationAccuracy;
        ac(lr) = x;
        
        
    end

[val,in] = max(ac);
sprintf('%.15g  is the best accuracy and was achieved with the learning rate %.15g for the optimizer %s',val,lrs(:,in), opts{opt})
best_lr = lrs(:,in);



% working with different mini batch sizes

mean_val = zeros(size(mbatches));
times = zeros(size(mbatches));
for mb = 1:length(mbatches)
    tic
modelOptions = trainingOptions(opts{:,opt}, ...
        'InitialLearnRate',lr, ...
        'MaxEpochs',epochs, ... 
        'ValidationData',ValidationSet, ...
        'ValidationFrequency',val_freq, ...
        'MiniBatchSize',mbatches(mb), ...
        'Verbose',false);
        %[net, results] = trainNetwork(TrainSet,NN_layers,modelOptions);
        f = @() trainNetwork(TrainSet,NN_layers,modelOptions);
        time = timeit(f);
        %x = results.ValidationAccuracy;
       % x(find(isnan(x)))=[];
        %mean_val(mb) = mean(x);
        times(mb) = time;
    toc
end
%[val,in] = max(mean_val);
[val,in] =min(times);
sprintf('%.15g  is the fastest training time eith a minibatchsize of  minibatchsize = %.15g with the learning rate %.15g and the optimizer %s',val,mbatches(in), best_lr, opts{opt})
end

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
