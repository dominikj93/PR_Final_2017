function a = my_rep_reshape(m, size_cell)

class_size = size(m, 1) / 10;

% interpolation method for resize ('bicubic', 'bilinear' or 'nearest');
resize_type = 'bicubic'; 
% size after resizing
size_row = size_cell;
size_col = size_cell;

% iterate over all data
for ii = 0:9 % number (class)
    for jj = 1:class_size % iterate over all items in each class
        idx = class_size * ii + jj; % index understandable for matlab
        proc_num = m(idx); % load single number for preprocessing
        im_num_org = data2im(proc_num); % convert PRTools datafile to image
%        im_num_org = skew_correction(im_num_org); % correct skewness
        im_num = im_num_org ...
          * im_resize([],[size_row size_col], resize_type); % size of images 
        
        a(idx,:) = im_num(:); %#ok<AGROW>
        labels{idx} = strcat('digit_',num2str(ii)); %#ok<AGROW> % make labels
    end
end

a = prdataset(a, labels); % output as dataset