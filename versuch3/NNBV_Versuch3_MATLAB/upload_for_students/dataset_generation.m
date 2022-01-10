%% Generation of dataset.
% pseudo random mode combination 

%%  set parameters 
number_of_modes = 3;    %option: 3 or 5
number_of_data = 10000;
image_size = 32;    % resolution 32x32

trainsize = 3300

%% generation of complex mmode weights and label vector - Aufgabe 1


% 1. create random amplitude weights. The weights of amplitude should be normalized.

rho=rand(number_of_data,number_of_modes);
rho_n=rho/norm(rho);


% 2. create random phase amplitude. (Using realtive phase difference)

phase = rand(number_of_data,1);
phase = phase/norm(phase); %normalize

phase = phase - phase(1); % make phase relative to first mode (phase difference)


% 3. complex mode weights vector

[weights_vector,label_vector] = mode_weights_generation(number_of_modes,number_of_data);

% 4. normalize cos(phase) to (0,1)

cos_n = normalization(cos(phase),0,1);

% 5. combine amplitude and phase into a label vector (1,2N-1)

label_vector = rho_n.*exp(1i*cos_n);


[weights_vector,label_vector] = mode_weights_generation(number_of_modes,number_of_data);

% 6. split complex mode weights vector and label vector into Training, validation and test set. 
trainset_weights = weights_vector(1:floor(number_of_data/3),:);
validationset_weights = weights_vector(floor(number_of_data/3)+1:floor(2*number_of_data/3),:);
testset_weights = weights_vector(floor(2*number_of_data/3)+1:end,:);

trainset_labels = label_vector(1:floor(number_of_data/3),:);
validationset_labels = label_vector(floor(number_of_data/3)+1:floor(2*number_of_data/3),:);
testset_labels = label_vector(floor(2*number_of_data/3)+1:end,:);

%% create image data - Aufgabe 2 
% use function mmf_build_image()

trainset = mmf_build_image(number_of_modes,image_size,trainsize, trainset_weights);
valset = mmf_build_image(number_of_modes,image_size,trainsize,validationset_weights);
testset = mmf_build_image(number_of_modes,image_size,trainsize,testset_weights);

%% save dataset
save trainset.mat trainset
save valset.mat valset
save testset.mat testset

