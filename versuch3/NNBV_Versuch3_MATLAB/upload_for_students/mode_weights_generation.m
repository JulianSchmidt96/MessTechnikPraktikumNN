function [mode_weights_vector,Label_vector] = mode_weights_generation(number_of_modes,number_of_random_data)
%% random amplitude value
weights_amp_ori = rand(number_of_random_data, number_of_modes);
amp_Norms = sqrt(sum( conj(weights_amp_ori).*weights_amp_ori, 2));
weights_amp = weights_amp_ori./amp_Norms; %% the weights of amplitude should be normalized.
weights_amp = round(weights_amp,5);  %% round to 4 digits to the right of the decimal point.

%% random phase value

weights_phase_ori = 2*pi*rand(number_of_random_data, number_of_modes-1);
weights_phase_1mode_zero = cat(2,zeros(number_of_random_data,1),weights_phase_ori);

original_mode_weights = cat(2,weights_amp,weights_phase_1mode_zero); %% Concatenate arrays

%% normalization of phase value, in range (0,2*pi)

cos_phase = cos(original_mode_weights(:,(number_of_modes+2):(2*number_of_modes)));
cos_phase_normalization = normalization(cos_phase,0,1);
Label_vector = cat(2,weights_amp,cos_phase_normalization);

%% create complex vectors

mode_weights_vector = weights_amp.*exp(1i*weights_phase_1mode_zero);

end

