load('train.mat')
load('validation.mat')
load('vocabulary.mat')
addpath('liblinear-2.11/windows/')

[trainInd,valInd,testInd] = dividerand(18092,0.99,0.1,0.0);

trainingX = X_train_bag(trainInd,:);
trainingY = Y_train(trainInd,:);
validationX = X_train_bag(valInd,:);
validationY = Y_train(valInd,:);

predictions = predict_labels_knn_jank(trainingX, trainingY, validationX);
% size(predictions)
% size(validationY)
score = performance_measure(full(predictions), full(validationY))

function [Y_hat] = predict_labels_knn_jank(trainingX, trainingY, validationX)
    
    k = 15;
    cost = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
    
    knn_model = fitcknn(trainingX, trainingY, 'NumNeighbors',k, 'Cost', cost);

    Y_hat = predict(knn_model, validationX);
end