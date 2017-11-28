load('train.mat')
load('validation.mat')
load('vocabulary.mat')
addpath('liblinear-2.11/windows/')
t = cputime;
[trainInd,valInd,testInd] = dividerand(18092,0.7,0.1,0.2);

trainingX = X_train_bag(trainInd,:);
trainingY = Y_train(trainInd,:);
validationX = X_train_bag(valInd,:);
validationY = Y_train(valInd,:);

predictions = predict_labels_jank(trainingX, trainingY, validationX, 1);
score = performance_measure(full(predictions), full(validationY))

% for k = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50, 100]
%     k
%     predictions = predict_labels_jank(training, Y_train, X_test_bag, k);
%     score = performance_measure(full(predictions), full(Y_train))
% end
% e = cputime - t;

function features = sparse_PCA(X, k)
    [U,S,V] = svds(X, k);
    features = U * S;
end

function [Y_hat] = predict_labels_jank(training_bag, training_labels, X_test_bag, k)
%     reduced_train_bag = sparse(sparse_PCA(training_bag, k));
    cost = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
%     ensemble_model = fitcnb(training_bag,training_labels,'Cost',cost);
    svm_model = train(full(training_labels), sparse(training_bag),'Cost',cost);
%     ensemble_model = fitcnb(reduced_train_bag,training_labels,'Cost',cost);
%     ensemble_model = fitensemble(reduced_train_bag,training_labels,'AdaBoostM2',500,'tree','Cost',cost);
%     SVM_model = fitensemble(reduced_train_bag,training_labels,'AdaBoostM2',10,'tree');
    disp('finished training');
%     reduced_test_bag = full(sparse_PCA(X_test_bag, k));
%     Y_hat = predict(ensemble_model, reduced_test_bag);
    Y_hat = predict(ones(size(X_test_bag,1),1), sparse(X_test_bag), svm_model);
% Inputs:   X_test_bag     nx9995 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text


% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)

end
