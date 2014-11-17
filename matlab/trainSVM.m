%% init and train

% We're gonna need a lot of memory...
clear all;

% Point this to the folder where you keep the data (1 subject for now)
folder = '~/Projects/UAB-Seizure Prediction/Dog_1/';

% What files to use
preictal_files = dirPattern([folder '*preictal*.mat']);
interictal_files = dirPattern([folder '*interictal*.mat']);
% test_files = dirPattern([folder '*test*.mat']);

% Change this to use different subsets (lazy/impatient mode on)
test_files = [preictal_files(21:24); interictal_files(21:24)];
preictal_files = preictal_files(1:20);
interictal_files = interictal_files(1:20);

% Important! Window size, # bands, etc
settings = loadjson('./settings.json');

% Init stuff
train_files = [preictal_files; interictal_files];
ntrain = numel(train_files);

% Process first recording to know what we should expect
filename = train_files{1};
bands = processSample([folder filename], settings);
[nch, nbands, nwin] = size(bands);

% Preallocate training matrix and store this recording
train_matrix = zeros(nch * nbands, nwin * ntrain);
train_matrix(:,1:nwin) = reshape(bands, [nch * nbands, nwin]);

% Iterate over recordings (10-min long) and generate feature vectors
for i = 2:ntrain

    % Load, split into windows, compute frequency bands and store
    filename = train_files{i};
    bands = processSample([folder filename], settings);
    [nch, nbands, nwin] = size(bands);
    wins = (i-1)*nwin+1 : i*nwin;
    train_matrix(:,wins) = reshape(bands, [nch*nbands, nwin]);

end

% Generate output class vector (assuming train_files is a row cell)
output = repmat(cellfun(@(x)isempty(strfind(x,'interictal')), train_files), nwin, 1);
output = double(output(:));

% -s svm_type : set type of SVM (default 0)
%     0 -- C-SVC
%     1 -- nu-SVC
%     2 -- one-class SVM
%     3 -- epsilon-SVR
%     4 -- nu-SVR
% -t kernel_type : set type of kernel function (default 2)
%     0 -- linear: u'*v
%     1 -- polynomial: (gamma*u'*v + coef0)^degree
%     2 -- radial basis function: exp(-gamma*|u-v|^2)
%     3 -- sigmoid: tanh(gamma*u'*v + coef0)
% -d degree : set degree in kernel function (default 3)
% -g gamma : set gamma in kernel function (default 1/num_features)
% -r coef0 : set coef0 in kernel function (default 0)
% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
% -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
% -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
% -m cachesize : set cache memory size in MB (default 100)
% -e epsilon : set tolerance of termination criterion (default 0.001)
% -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
% -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
% -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)

out.svmParams = '-s 0 -t 1 -d 3 -h 0 -m 1000 -v 10';
out.model = svmtrain(output, train_matrix', out.svmParams);

saveData('model.mat', out);

%% test

ntest = numel(test_files);

% Iterate over recordings (10-min long) and generate feature vectors
for i = 1:ntest

    % Load, split into windows, compute frequency bands and store
    filename = test_files{i};
    bands = processSample([folder filename], settings);
    [nch, nbands, nwin] = size(bands);
    output = ones(nwin, 1) * isempty(strfind(filename,'interictal'));
    test_matrix = reshape(bands, [nch*nbands, nwin]);
    [pred, acc_p] = svmpredict(zeros(size(rawn,1),1), rawn, model.model);

end
