%% HERE RAW DATA IS PREPROCESSED 
function a = my_rep(m,size_cell,is_hog)
%%
% determine size of each class:
class_size = size(m, 1) / 10;

% get labels from the data
% se_closing = strel('disk',1);

% interpolation method for resize ('bicubic', 'bilinear' or 'nearest');
resize_type = 'bicubic'; 
% size after resizing
size_row = size_cell;
size_col = size_cell;
% cell size for HOG features
cell_size = size_row/4;
if is_hog == 4
    cell_size = size_row/3;
    cell_size = floor(cell_size);
end
% iterate over all data
for ii = 0:9 % number (class)
    for jj = 1:class_size % iterate over all items in each class
        
        idx = class_size * ii + jj; % index understandable for matlab
        proc_num = m(idx); % load single number for preprocessing
        im_num_org = data2im(proc_num); % convert PRTools datafile to image
        
        % Morphological operations
        im_num2 = bwmorph(im_num_org,'bridge'); % bridges unconnected pixels
        im_num3 = bwmorph(im_num2,'clean'); % removes isolated pixels
        im_num4 = im_num3; %#ok<NASGU> % bwmorph(im_num3,'thin',2);
        im_num5 = bwmorph(im_num3,'fill'); % fill single pixels
        %im_num6 = bwmorph(im_num5,'close',3); % performs morphological closing
        
        im_num7 = bwareaopen(+im_num5,30); % delete small objects (noise)
%         im_num7 = skew_correction(im_num7); % correct skewness
        im_num8 = double(im_num7);
        im_out = im_num8 ... 
          * im_resize([],[size_row size_col], 'bicubic'); % size of images 

%        manual_testing(im_num_org,im_num2,im_num3,im_num4,im_num5,im_num6,im_num7,im_num8,im_out);
        
        [featureVector, ~] = extractHOGFeatures(im_out,'CellSize',[cell_size cell_size]);
        hog_feat(idx,:) = featureVector(:); %#ok<AGROW,NASGU> % store HOG features
%         hog_im{idx} = hogVisualization; % store HOG visualization
        labels{idx} = strcat('digit_',num2str(ii)); %#ok<AGROW> % make labels
        
%         im_out = im_num8 ...
%           * im_box([],0,1) ... % add rows/columns to make images square 
%           * im_box([],1,0); % add rows/columns and keep image square

        a(idx,:) = im_out(:); %#ok<AGROW> % store output images
        
    end
end

a = prdataset(a, labels); % output as dataset
% if one wants to use HOG features instead of pixels please uncomment next line
if is_hog == 3
clear a; hog_f = double(hog_feat); a = prdataset(hog_f,labels);
end
%% Stuff below is only for testing
% [train_data,test_data] = gendat(processed_data,0.8);
% pca_im = pcam(a);

% figure;
% imshow(reshape(a(1,:),[size_row+2 size_col+2])); hold on;
% plot(hog_im{1});