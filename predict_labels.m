function [Y_hat] = predict_labels(X_test_bag, test_raw)
    load('train.mat')
    load('vocabulary.mat')
    
    SVM_model = fitcecoc(full(X_train_bag), Y_train);
    Y_hat = predict(SVM_model, X_test_bag)

% Inputs:   X_test_bag     nx9995 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text


% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)

end