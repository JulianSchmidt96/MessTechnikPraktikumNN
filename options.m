modelOptions = trainingOptions('adam', ...
        'InitialLearnRate',0.0001, ...
        'MaxEpochs',30, ... 
        'ValidationData',ValidationSet, ...
        'ValidationFrequency',val_freq, ...
        'Verbose',false);