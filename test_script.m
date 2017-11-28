load('train.mat')
load('validation.mat')
load('vocabulary.mat')
addpath('liblinear-2.11/windows/')

[trainInd,valInd,testInd] = dividerand(18092,0.95,0.05,0.0);

trainingX = X_train_bag(trainInd,:);
trainingY = Y_train(trainInd,:);
validationX = X_train_bag(valInd,:);
validationY = Y_train(valInd,:);

total_word_usage = sum(X_train_bag);
size(total_word_usage)
global good_words
good_words = total_word_usage > 2;

predictions = predict_labels_jank(trainingX, trainingY, validationX);
size(predictions)
size(validationY)
score = performance_measure(full(predictions), full(validationY))

function features = sparse_PCA(X, k)
    [U,S,V] = svds(X, k);
    features = U * S;
end

% I thought that excluding low-frequency words would reduce overfitting
% but now I think that the L1 regularization mostly takes care of this
function better_bag = get_good_words(bag)
    global good_words
%     better_bag = bag(:, good_words);
    better_bag = bag;
end

function [Y_hat] = predict_labels_jank(training_bag, training_labels, X_test_bag)

    reduced_train_bag = get_good_words(training_bag);
    cost = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
    logit_model = train(full(training_labels), reduced_train_bag,'-s 0 -c 0.35');
    reduced_test_bag = get_good_words(X_test_bag);
    
    ntrials = 500;
    nrows = size(X_test_bag,1);
    results = zeros(nrows, ntrials);
    
    disp('lol');
    for i = 1:500
        results(:, i) = predict(ones(size(X_test_bag,1),1), reduced_test_bag, logit_model);
    end
   
    probabilities = zeros(nrows,5);
    for i = 1:nrows
        probabilities(i,1) = nnz(results(i,:) == 1) / ntrials;
        probabilities(i,2) = nnz(results(i,:) == 2) / ntrials;
        probabilities(i,3) = nnz(results(i,:) == 3) / ntrials;
        probabilities(i,4) = nnz(results(i,:) == 4) / ntrials;
        probabilities(i,5) = nnz(results(i,:) == 5) / ntrials;
    end
    
    disp('lol2');
    
    output = zeros(nrows, 1);
    for i = 1:nrows
        [value,index]=max(probabilities(i,:));
        output(i) = index;
        disp('lol3');
    end
    size(output)
    Y_hat = output;
% Inputs:   X_test_bag     nx9995 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text
% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)

end
