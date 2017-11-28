load('train.mat')
load('validation.mat')
load('vocabulary.mat')
addpath('liblinear-2.11/windows/')

[trainInd,valInd,testInd] = dividerand(18092,0.7,0.1,0.2);

trainingX = X_train_bag(trainInd,:);
trainingY = Y_train(trainInd,:);
validationX = X_train_bag(valInd,:);
validationY = Y_train(valInd,:);

total_word_usage = sum(X_train_bag);
size(total_word_usage)
global good_words
good_words = total_word_usage > 30;

predictions = predict_labels_jank(trainingX, trainingY, validationX);
score = performance_measure(full(predictions), full(validationY))

function features = sparse_PCA(X, k)
    [U,S,V] = svds(X, k);
    features = U * S;
end

function better_bag = get_good_words(bag)
    global good_words
    better_bag = bag(:, good_words);
end

function [Y_hat] = predict_labels_jank(training_bag, training_labels, X_test_bag)

    reduced_train_bag = get_good_words(training_bag);
    cost = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
    svm_model = train(full(training_labels), reduced_train_bag,'Cost',cost);
    disp('finished training');
    reduced_test_bag = get_good_words(X_test_bag);
    Y_hat = predict(ones(size(X_test_bag,1),1), reduced_test_bag, svm_model);
% Inputs:   X_test_bag     nx9995 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text
% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)

end

function [Y_hat] = predict_labels_boosted_logit(training_bag, training_labels, X_test_bag)
    for m = 1:100
        cost = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
        logit_model = train(full(training_labels), training_bag,'Cost',cost);
        Y_hat = predict(ones(size(X_test_bag,1),1), X_test_bag, logit_model);
    end
end
