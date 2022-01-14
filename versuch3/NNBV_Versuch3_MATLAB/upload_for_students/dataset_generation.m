%% Generation of dataset.
% pseudo random mode combination 

%%  set parameters 
number_of_modes = 3;    %option: 3 or 5
number_of_data = 10000;
image_size = 32;    % resolution 32x32

trainsize = 3333;

%% generation of complex mmode weights and label vector - Aufgabe 1




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

trainset = struct;
trainset.images = mmf_build_image(number_of_modes,image_size,trainsize, trainset_weights),trainset_labels;
trainset.labels= trainset_labels;

valset = struct;
valset.images = mmf_build_image(number_of_modes,image_size,trainsize, validationset_weights),validationset_labels;  
valset.labels = validationset_labels;

testset = struct;
testset.images = mmf_build_image(number_of_modes,image_size,trainsize,testset_weights),testset_labels;
testset.labels = testset_labels;     

%% save dataset
save('trainset.mat', 'trainset');
save valset.mat valset
save testset.mat testset
 
 
