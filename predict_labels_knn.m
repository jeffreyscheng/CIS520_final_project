function [Y_hat] = predict_labels_knn(X_test_bag, test_raw)
    load('train.mat')
    load('validation.mat')
    load('vocabulary.mat')
    %addpath('liblinear-2.11/matlab/')
    
    k = 5;
    cost = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
    
    knn_model = fitcknn(X_train_bag, Y_train, 'NumNeighbors',k, 'Cost', cost);

    Y_hat = predict(knn_model, X_test_bag);
end