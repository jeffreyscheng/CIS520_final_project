function [Y_hat] = predict_labels_nb(X_test_bag, test_raw)
    load('train.mat')
    load('validation.mat')
    load('vocabulary.mat')
    %addpath('liblinear-2.11/matlab/')
    
    k = 15;
   
    
    % nonsparse_training(1:5)
    reduced_train_bag = full(sparse_PCA(X_train_bag, k));
    % reduced_train_bag = reduced_train_bag(:,1:1000)
    ClassNames = {'1', '2', '3', '4', '5'};
    training_labels = ClassNames(Y_train);
    cost.ClassNames = ClassNames;
    cost.ClassificationCosts = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
    ensemble_model = fitcnb(reduced_train_bag,training_labels,'Cost',cost);
    reduced_test_bag = full(sparse_PCA(X_test_bag, k));
    predicted = predict(ensemble_model, reduced_test_bag);
    Y_hat = str2double(predicted);
end