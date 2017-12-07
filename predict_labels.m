function [Y_hat] = predict_labels(X_test_bag, test_raw)
    load('train.mat')
    load('validation.mat')
    load('vocabulary.mat')
    addpath('liblinear-2.11/matlab/')

    reduced_train_bag = X_train_bag;
    cost = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
    logit_model = train(full(Y_train), reduced_train_bag,'-s 0 -c 0.25');
    reduced_test_bag = X_test_bag;
    
    ntrials = 500;
    nrows = size(X_test_bag,1);
    results = zeros(nrows, ntrials);
    
    for i = 1:2000
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
    
    exp_costs = probabilities * cost;
%    output = zeros(nrows, 1);
%     for i = 1:nrows
%         [value,index]=max(probabilities(i,:));
%         output(i) = index;
%         disp('lol3');
%     end
%     size(exp_costs)
    [~,output] = min(exp_costs,[],2);
%     size(output)
    Y_hat = output;
% Inputs:   X_test_bag     nx9995 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text


% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)

end