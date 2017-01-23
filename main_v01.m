%% Patter Recognition - Final Assignment
% Pall Bjornsson 
% Dominik Jargot 4633504

%% CLEAR WORKSPACE
% This file should be executed very frst, this section cleans workspace
% and prepares raw data
clear
close all
clc
prwaitbar off
prmemory(80000000);


%% LOAD DATA
% This section is loading data, please determine following parameters:
nist_data = prnist(0:9,1:1000);

% choose how many objects for each digit is wanted in first scenario, i.e.
% 1) the pattern recognition system is trained once, and then applien in
% the field (choose num_digit between 200 and 1000)
num_dense_digit = 200;
% choose how many objects for each digit is wanted in second scenario, i.e.
% 2) the pattern recognition system is trained for each batch of cheques to
% be processed (choose num_digit between 1 and 10)
num_sparse_digit = 15;

%%
for data_set = 1:5

    % load randomly the dataset for 1st scenario
    ratio_dense = num_dense_digit/1000;
    dense_data = gendat(nist_data, ratio_dense);
    
    % load randomly the dataset for 2nd scenario
    ratio_sparse = num_sparse_digit/1000;
    sparse_data = gendat(nist_data, ratio_sparse);
    
    % show original data
    % show(sparse_data);
    % show(dense_data);
    %%
    for is_preprocessing = 1:3 % 1 - proccesed, 2 - only reshaped, 3 - hog, 4 - hog smaller
        size_idx = 1;
        for size_im = 12:4:28 % iterate over sizes of image (16,32,48)
            pca_idx = 1;
            %% PREPROCESSING
            if is_preprocessing == 1 || is_preprocessing == 3 || is_preprocessing == 4
                sd_pre = my_rep(sparse_data,size_im,is_preprocessing);
                % dd_pre = my_rep(dense_data,size_im);
            elseif is_preprocessing == 2
                sd_pre = my_rep_reshape(sparse_data,size_im);
                % dd_pre = dense_data;
            end
            disp('Data proceeded, now prepare for the long part! There are 9 classifiers to validate!');
            %%
            % sd_pre = im_features(sd_pre); % extract 24 features
            % dd_pre = im_features(sd_pre); % extract 24 features
            for jj = 0.88:0.03:0.97 % iterate over PCA (0.7, 0.8, 0.9)
                                
                pca_ratio = jj; % set PCA ratio (how much variance you want to keep)
                tic
%                 disp('Train first classifier (knnc3):');
                [error_knnc3(data_set,is_preprocessing,size_idx,pca_idx)] = prcrossval(sd_pre,scalem([],'c-variance')*pcam([],pca_ratio)*knnc([],3),3);
%                 disp('Train first classifier (knnc4):');
                [error_knnc4(data_set,is_preprocessing,size_idx,pca_idx)] = prcrossval(sd_pre,scalem([],'c-variance')*pcam([],pca_ratio)*knnc([],4),3);
%                 disp('Train first classifier (knnc5):');
                [error_knnc5(data_set,is_preprocessing,size_idx,pca_idx)] = prcrossval(sd_pre,scalem([],'c-variance')*pcam([],pca_ratio)*knnc([],5),3);
%                 disp('Train second classifier (fisherc):');
%                 [error_fisherc(data_set,is_preprocessing,size_idx,pca_idx)] = prcrossval(sd_pre,scalem([],'variance')*pcam([],pca_ratio)*fisherc,5);
%                 disp('Train third classifier (loglc):');
                [error_loglc(data_set,is_preprocessing,size_idx,pca_idx)] = prcrossval(sd_pre,scalem([],'c-variance')*pcam([],pca_ratio)*loglc,3);
%                 disp('Train fourth classifier (adaboostc):');
%                 [error_adaboostc(data_set,is_preprocessing,size_idx,pca_idx)] = prcrossval(sd_pre,scalem([],'variance')*pcam([],pca_ratio)*adaboostc,5);
%                 disp('Train fifth classifier (parzenc):');
                [error_parzenc(data_set,is_preprocessing,size_idx,pca_idx)] = prcrossval(sd_pre,scalem([],'c-variance')*pcam([],pca_ratio)*parzenc,3);
%                 disp('Train sixth classifier (bpxnc):')
%                 [error_bpxnc(data_set,is_preprocessing,size_idx,pca_idx)] = prcrossval(sd_pre,scalem([],'variance')*pcam([],pca_ratio)*bpxnc([],300),5);
%                 disp('Train seventh classifier (pksvc):')
                [error_pksvc(data_set,is_preprocessing,size_idx,pca_idx)] = prcrossval(sd_pre,scalem([],'c-variance')*pcam([],pca_ratio)*pksvc,3);
%                 disp('Train eight classifier (quadrc):')
%                 [error_quadrc(data_set,is_preprocessing,size_idx,pca_idx)] = prcrossval(sd_pre,scalem([],'variance')*pcam([],pca_ratio)*quadrc,5);
%                 disp('Train ninth classifier (nmc):')
                [error_nmc(data_set,is_preprocessing,size_idx,pca_idx)] = prcrossval(sd_pre,scalem([],'c-variance')*pcam([],pca_ratio)*nmc,3);
%                 disp('Train tenth classifier (nmsc):')
                [error_nmsc(data_set,is_preprocessing,size_idx,pca_idx)] = prcrossval(sd_pre,scalem([],'c-variance')*pcam([],pca_ratio)*nmsc,3);
%                 disp('Train eleventh classifier (svc):')
                [error_svc(data_set,is_preprocessing,size_idx,pca_idx)] = prcrossval(sd_pre,scalem([],'c-variance')*pcam([],pca_ratio)*svc,3);
                toc
                                
                fprintf('%d / 360\n',36*(data_set-1)+(is_preprocessing-1)*12+(size_idx-1)*4+pca_idx);
                pca_idx = pca_idx + 1;
            end
            size_idx = size_idx + 1;
        end
    end
end
disp('The end! Check the results! AND SAVE THEM!!!!!')

% %%
% d=prdataset(c); %working dataset
% nrep = 4; %number of repetitions for the classifier evaluations.
% [Train,Test]=gendat(d,0.5); %data set split
% 
% dSc2=Train*scalem(Train,'variance');%scaling of the features
% [W2,frac2]=pcam(dSc2,32); %PCA on the original dataset. 
% f=Train*W2; %projecting d on W1 for dim reduction
% PznLC=parzenc(f);%parzen classifier
% 
% display('Parzen, PCA 32');
% testc(Test*W2,{PznLC});
% [x,z] = gendat(sparse_data,0.8);
% u=scalem([],'variance')*pcam([],0.9)*parzenc;
% w = x*u;
% z*w*testc;