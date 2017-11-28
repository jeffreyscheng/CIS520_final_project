load('train.mat')
load('validation.mat')
load('vocabulary.mat')

t = cputime;
[trainInd,valInd,testInd] = dividerand(18092,0.5,0.1,0.0);

trainingX = X_train_bag(trainInd,:);
trainingY = Y_train(trainInd,:);
validationX = X_train_bag(valInd,:);
validationY = Y_train(valInd,:);

predictions = predict_labels_jank(trainingX, trainingY, validationX);
validationY(1:10,:)
predictions(1:10,:)
score = performance_measure(predictions, validationY)
e = cputime - t
function features = sparse_PCA(X)
%     [~,~,PC] = svds(X,100);
%     mu = mean(X);
%     S = sparse(size(X,1),100);
%     for i=1:size(X,1)
%         S(i,:) = (X(i,:)-mu)*PC;
%     end
%     features = S;
%     2

% 0.1, 0.1, 1000 -> 1.6306
% 0.3, 0.1, 500 -> 1.7690
    [U,S,V] = svds(X, 1000);
    features = U * S;
end

function [Y_hat] = predict_labels_jank(training_bag, training_labels, X_test_bag)
    % nonsparse_training = full(training_bag);
    % size(nonsparse_training)
    % nonsparse_training(1:5)
    reduced_train_bag = full(sparse_PCA(training_bag))
    % reduced_train_bag = reduced_train_bag(:,1:1000)
    ClassNames = {'1', '2', '3', '4', '5'};
    training_labels = ClassNames(training_labels);
    cost.ClassNames = ClassNames;
    cost.ClassificationCosts = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
    ensemble_model = fitensemble(reduced_train_bag,training_labels,'AdaBoostM2',10,'tree','Cost',cost);
%     SVM_model = fitensemble(reduced_train_bag,training_labels,'AdaBoostM2',10,'tree');
    disp('finished training');
    reduced_test_bag = full(sparse_PCA(X_test_bag));
    % reduced_test_bag = reduced_test_bag(:,1:1000);
    Y_hat = str2double(predict(ensemble_model, reduced_test_bag));

% Inputs:   X_test_bag     nx9995 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text


% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)

end
