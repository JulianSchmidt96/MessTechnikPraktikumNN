start_mbatch = floor(255/6);
end_mbatch = floor(255/6) * 6;

mbatches = addedarray(start_mbatch, end_mbatch, start_mbatch);



mean_val = zeros(size(mbatches));
times = zeros(size(mbatches));
for mb = 1:length(mbatches)
       
modelOptions = trainingOptions(opts{:,opt}, ...
        'InitialLearnRate',lr, ...
        'MaxEpochs',epochs, ... 
        'ValidationData',ValidationSet, ...
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




function arr = addedarray(startval, endval, stepadd)
    i=startval;
    k = 1;
    
    arr = [];

    while i~=endval
        arr(k) = i;
       i = i + stepadd;
       k = k + 1;
    end
