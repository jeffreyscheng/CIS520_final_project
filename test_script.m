load('train.mat')
load('validation.mat')
load('vocabulary.mat')

trainingX = X_train_bag(1:15000,:);
trainingY = Y_train(1:15000,:);
validationX = X_train_bag(15001:18092,:);
validationY = Y_train(15001:18092,:);

predictions = predict_labels_jank(trainingX, trainingY, validationX);
score = performance_measure(predictions, validationY)

function features = sparse_PCA(X)
    [~,~,PC] = svds(X,1000);
    mu = mean(X);
    S = sparse(size(X,1),1000);
    for i=1:size(X,1)
        S(i,:) = (X(i,:)-mu)*PC;
    end
    features = S;
end

function [Y_hat] = predict_labels_jank(training_bag, training_labels, X_test_bag)
    % nonsparse_training = full(training_bag);
    % size(nonsparse_training)
    % nonsparse_training(1:5)
    reduced_train_bag = sparse_PCA(training_bag);
    % reduced_train_bag = reduced_train_bag(:,1:1000);
    cost = [0,3,1,2,3;4,0,2,3,2;1,2,0,2,1;2,1,2,0,2;2,2,2,1,0];
    SVM_model = fitctree(reduced_train_bag,training_labels,'Cost',cost);
    % SVM_model.costs = cost;
    reduced_test_bag = sparse_PCA(X_test_bag);
    % reduced_test_bag = reduced_test_bag(:,1:1000);
    Y_hat = predict(SVM_model, reduced_test_bag);

% Inputs:   X_test_bag     nx9995 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text


% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)

end
