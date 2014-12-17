function [accuracy, results, models] = trainSVM(settings)
% Test an SVM system with cross-validation
% Input: structure with settings (path, etc) as in settings.json
% Output:
%     - accuracy: array with the accuracy for each fold
%     - results: cell with a recording score arrays for each fold
%          -1: false negative
%           0: hit
%           1: false positive
%     - models: cell with the model, plus metadata, for each fold

    % Point this to the folder where you keep the data (1 subject for now)
    folder = settings.data.path;

    % What files to use
    preictal_files = dirPattern([folder filesep '*preictal*.mat']);
    interictal_files = dirPattern([folder filesep '*interictal*.mat']);

    % Change this to use different subsets (lazy/impatient mode on)
    if settings.data.preictal > 0
        preictal_files = preictal_files(1:settings.data.preictal);
    end
    if settings.data.interictal > 0
        interictal_files = interictal_files(1:settings.data.interictal);
    end

    % Build dataset
    labels = [ones(length(preictal_files), 1); zeros(length(interictal_files), 1)];
    files  = [preictal_files(:); interictal_files(:)];
    nfiles = numel(files);

    % Cross-validation partitions
    cv = cvpartition(labels, 'kfold', settings.crossvalidation.folds);

    accuracy = zeros(cv.NumTestSets, 1);
    results = {};
    models = {};

    for fold = 1:cv.NumTestSets

        train_files = files(training(cv, fold));
        train_labels = labels(training(cv, fold));
        test_files = files(test(cv, fold));
        test_labels = labels(test(cv, fold));
        ntrain = length(train_files);
        ntest = length(test_files);

        %% TRAIN

        % Process first recording to know what we should expect
        filename = train_files{1};
        bands = processSample([folder filesep filename], settings);
        [nch, nbands, nwin] = size(bands);

        % Preallocate training matrix and store this recording
        train_matrix = zeros(nch * nbands, nwin * ntrain);
        train_matrix(:,1:nwin) = reshape(bands, [nch * nbands, nwin]);

        % Iterate over recordings (10-min long) and generate feature vectors
        for i = 2:ntrain

            % Load, split into windows, compute frequency bands and store
            filename = train_files{i};
            bands = processSample([folder filesep filename], settings);
            [nch, nbands, nwin] = size(bands);
            wins = (i-1)*nwin+1 : i*nwin;
            train_matrix(:,wins) = reshape(bands, [nch*nbands, nwin]);

        end

        % Generate output class vector
        % 1 is preictal (seizure), 0 interictal (no seizure)
        output_train = double(repmat(train_labels(:)', nwin, 1));
        output_train = output_train(:);

        disp('Matrix ready. Training model...');

        % SVM parameters are loaded from the JSON structure with the SVMPARSE program.
        % Additional, hard-coded parameters can be specified by concatenation:
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
        % -v nfolds: cross-validation with nfolds. Output score instead of model.

        out.svmParams = [svmparse(settings.svm)]; %'-w0 0.7 -w1 0.3'];
        out.settings = settings;
        out.train_files = train_files;
        out.model = svmtrain(output_train, train_matrix', out.svmParams);

        models{fold} = out;

        %% TEST

        ntest = numel(test_files);

        % Iterate over recordings (10-min long) and generate feature vectors
        for i = 1:ntest

            % Load, split into windows, compute frequency bands and store
            filename = test_files{i};
            bands = processSample([folder filesep filename], settings);
            [nch, nbands, nwin] = size(bands);
            test_matrix = reshape(bands, [nch*nbands, nwin]);
            output_test = ones(nwin,1) * test_labels(i);
            [pred{i}, acc_p{i}] = svmpredict(output_test, test_matrix', out.model);

        end

        % Very naive method, without kalman filters.
        % Just predict seizure if most windows have been predicted as seizure

        test_labels_cell = num2cell(test_labels);
        result = cellfun(@(pred,truth) ...
            ((mean(pred) < settings.threshold) - truth), ...
            pred(:), test_labels_cell(:));
        accuracy(fold) = mean(abs(result));
        results{fold} = result;

        disp('-----------------------------');
        fprintf('Accuracy (fold %d): %f\n', fold, accuracy(fold));
        disp('-----------------------------');

    end

    disp('=============================');
    fprintf('Accuracy (total): %f\n', mean(accuracy));
    disp('=============================');

end
