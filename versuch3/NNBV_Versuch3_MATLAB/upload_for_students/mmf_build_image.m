function [Image_data] = mmf_build_image(number_of_modes,image_size,number_of_data, complex_weights_vector)
%% load complex mode distribution
% load the complex distrbutions 


%% create images
% define a variable for Image data with dimension (image size, image size, 1, n)
fprintf("Start to generate the mode distribution......\n");
Image_data = zeros(image_size,image_size,1,number_of_data);


mmf =  load('/home/schmijul/MessTechnikPraktikumNN/MessTechnikPraktikumNN/versuch3/NNBV_Versuch3_MATLAB/upload_for_students/mmf_3modes_32.mat');


for index_number=1:number_of_data
    % 1. define a variable for single image with resolution (image size,image size)

    single_image = zeros(image_size,image_size);
    
    % 2. generation of complex field distribution 

    for index_mode=1:number_of_modes
        single_image = single_image + complex_weights_vector(index_number,index_mode) * mmf.mmf_3modes_32(:,:,index_mode);
    end

    
    % 3. abstract Amplitude distribution 

    rho_max = max(max(abs(single_image)));
    single_image = single_image/rho_max;

    % 4. normalization the amplitude distribution to (0,1)

    %    using normalization(image, minValue, maxValue)
    
   % imagesc(abs(single_image));
   % drawnow;
   % pause;

    Image_data(:,:,1,index_number) = single_image;
end
fprintf("The image data has been generated.\n");

end

