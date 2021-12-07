%% Lade nicht augmentierte Trainingsdaten
load('DATA_MMF_16_2.mat');            % Trainingsdaten
load('MMF_Param_16.mat');              % Parameter der MMF
N = size(XTrain,4);                 % Anzahl Trainingsbilder
r = size(XTrain,1);                 % Auflösung Trainingsbilder=[r,r]

%% Neue Trainingsdaten
XTrain_aug = zeros(r,r,1,2*N);
YTrain_aug = zeros(r,r,1,2*N);

% %%
% I_px = size(XTrain,1)
% imagesize = [I_px I_px 1]
% imageAugmenter = imageDataAugmenter( ...
%     'RandRotation',[0 360]);
% 
% auimds = augmentedImageDatastore(imagesize,XTrain,YTrain,"DataAugmentation",imageAugmenter)

%%
tform = randomAffine2d('Rotation',[-45 45]);

% imAugmented = imwarp(imOriginal,tform,'OutputView',outputView);  

%% Data Augmentation (2 neue Bilder pro Trainingsbild)
for i1=1:N
    original_image = XTrain(:,:,:,i1);
    
   % Data Augmentation 1
     aug_image =imwarp(original_image,tform);  
    
    [XTrain_aug(:,:,:,i1), YTrain_aug(:,:,:,i1)] = mmf(aug_image,r,M_T,modes_n);
    
   
end

%% Save Augmented Training Data
XTrain = cat(4,XTrain,XTrain_aug);
YTrain = cat(4,YTrain,YTrain_aug);
save('DATA_MMF_16_aug_2.mat','XTrain','YTrain','XValid','YValid','XTest','YTest');