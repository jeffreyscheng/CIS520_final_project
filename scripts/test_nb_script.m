load('train.mat')
load('validation.mat')
load('vocabulary.mat')
addpath('liblinear-2.11/windows/')

[trainInd,valInd,testInd] = dividerand(18092,0.99,0.1,0.0);

trainingX = X_train_bag(trainInd,:);
trainingY = Y_train(trainInd,:);
validationX = X_train_bag(valInd,:);
validationY = Y_train(valInd,:);

predictions = predict_labels_nb_jank(trainingX, trainingY, validationX);
% size(predictions)
% size(validationY)
score = performance_measure(full(predictions), full(validationY))

function [Y_hat] = predict_labels_nb_jank(trainingX, trainingY, validationX)
    %addpath('liblinear-2.11/matlab/')
    
    k = 15;
   
    
    % nonsparse_training(1:5)
    reduced_train_bag = full(sparse_PCA(trainingX, k));
    % reduced_train_bag = reduced_train_bag(:,1:1000)
    ClassNames = {'1', '2', '3', '4', '5'};
    training_labels = ClassNames(trainingY);
    cost.ClassNames = ClassNames;
    cost.ClassificationCosts = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
    ensemble_model = fitcnb(reduced_train_bag,training_labels,'Cost',cost);
    reduced_test_bag = full(sparse_PCA(validationX, k));
    predicted = predict(ensemble_model, reduced_test_bag);
    Y_hat = str2double(predicted);
end