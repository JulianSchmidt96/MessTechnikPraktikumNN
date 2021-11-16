% fetch data-set

[x_train,~,y_train] = digitTrain4DArrayData;
[x_val,~,y_val] = digitTest4DArrayData;
%% Build the Model

inputShape = [size(x_train,1,2) 1];


layers = [
    imageInputLayer(inputShape)
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer

    averagePooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    averagePooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    dropoutLayer(0.2)
    fullyConnectedLayer(1)
    regressionLayer];
%% Hyperparams

opt = 'adam';
lr = 0.1;
epochs = 2;
miniBatchSize  = 128; % size of data used per iteration


validationFrequency = floor(numel(y_train)/miniBatchSize);




%%
start_lr = 10^-6;
end_lr = 10^-1;
stepmulti = 10;
lrs = fetchlearningrates(start_lr, end_lr, stepmulti);
%%
val_loss = zeros(epochs + 1, length(lrs));

k=1;

for lr = 1:length(lrs)
    modelOptions = trainingOptions(opt, ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',epochs, ...
    'InitialLearnRate',lr, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{x_val, y_val}, ...
    'ValidationFrequency',validationFrequency, ...
    'Verbose',false);
    [net, history] = trainNetwork(x_train,y_train,layers,modelOptions);
    
    x = history.ValidationLoss;
    x(find(isnan(x)))=[]
    val_loss(:,lr) = x
    
    k = k + 1;
end
%%
plot(val_loss)


legendStrings = "lr = " + string(lrs);
legend(legendStrings)

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
