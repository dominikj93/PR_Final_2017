function proccesed_im = skew_correction(input_im)

% removing slant
input_im = double(input_im); % make data readable for im_box
%input_im = im_box(input_im,[10,10,10,0]);
M = im_moments(input_im,'central');       % central moments
theta = atan(2*M(3)/(M(1)-M(2)));   % skewness 
if theta<0 % Correction of failures in skew computation
    theta = 0.5*pi;
end
A = [1 0 0; sin(0.5*pi-theta) cos(0.5*pi-theta) 0;0 0 1];
tform = affine2d(A);
proccesed_im = imwarp(input_im,tform,'nearest');

end