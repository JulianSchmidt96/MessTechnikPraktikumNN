function [Image_data] = mmf_build_image(number_of_modes,image_size,number_of_data, complex_weights_vector)
%% load complex mode distribution
% load the complex distrbutions 


%% create images
% define a variable for Image data with dimension (image size, image size, 1, n)
Image_data = zeros(image_size,image_size,1,number_of_data);
fprintf("Start to generate the mode distribution......\n");

for index_number=1:number_of_data
    % 1. define a variable for single image with resolution (image size,image size)
    single_image = zeros(image_size,image_size);
    
    % 2. generation of complex field distribution 

    
        % 2.1 define a variable for complex field distribution with resolution (image size,image size)
        complex_field_distribution = zeros(image_size,image_size);


       
        
        % 2.2 generation of complex field distribution 
        for index_mode=1:number_of_modes
            
            complex_field_distribution = complex_field_distribution + complex_weights_vector(index_mode)*exp(1i*2*pi*rand(image_size,image_size));
        end
       
        
        % 2.3 add the complex field distribution to the image
        single_image = single_image + abs(complex_field_distribution).^2;

        
    end

    % 3. abstract Amplitude distribution 
    Image_data(:,:,1,index_number) = single_image;
    % 4. normalization the amplitude distribution to (0,1)


    %    using normalization(image, minValue, maxValue)
    Image_data(:,:,1,index_number) = normalization(single_image,0,1);
    

    
    
end
fprintf("The image data has been generated.\n");

end

