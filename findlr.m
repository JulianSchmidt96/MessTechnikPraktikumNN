start_lr = 10^-6;
end_lr = 10^-1;

lrs = multiplicatedarray(start_lr, end_lr, 10);
epochs = 10;

val_freq = 30;



opts = {'adam','sgdm'};
optimizier = opts{1};
for opt =1:length(opts)

    ac = zeros(size(lrs));
    for lr = 1:length(lrs)
        modelOptions = trainingOptions(opts{:,opt}, ...
        'InitialLearnRate',lr, ...
        'MaxEpochs',epochs, ... 
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
