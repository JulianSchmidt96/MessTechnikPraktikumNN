%% Generation of dataset.
% pseudo random mode combination 

%%  set parameters 
number_of_modes = 3;    %option: 3 or 5
number_of_data = 10000;
image_size = 32;    % resolution 32x32
 
%% generation of complex mode weights and label vector - Aufgabe 1


% 1. create random amplitude weights. The weights of amplitude should be normalized.
rho=rand(1,number_of_data);
rho_n=rho/norm(rho);


% 2. create random phase amplitude. (Using realtive phase difference)
phase = rand(number_of_data,1);
phase = phase/norm(phase); %normalize

phase = phase - phase(1); % make phase relative to first mode (phase difference)


% 3. complex mode weights vector

weights_vector = rho_n.*exp(1i*phase);

% 4. normalize cos(phase) to (0,1)

cos_n = normalization(cos(phase),0,1);

% 5. combine amplitude and phase into a label vector (1,2N-1)
label_vector = rho_n.*exp(1i*cos_n);

% 6. split complex mode weights vector and label vector into Training, validation and test set. 
trainset_weights = weights_vector(1:floor(number_of_data/3),:);
validationset_weights = weights_vector(floor(number_of_data/3)+1:floor(2*number_of_data/3),:);
testset_weights = weights_vector(floor(2*number_of_data/3)+1:end,:);

trainset_labels = label_vector(1:floor(number_of_data/3),:);
validationset_labels = label_vector(floor(number_of_data/3)+1:floor(2*number_of_data/3),:);
testset_labels = label_vector(floor(2*number_of_data/3)+1:end,:);

%% create image data - Aufgabe 2 
% use function mmf_build_image()


%% save dataset

