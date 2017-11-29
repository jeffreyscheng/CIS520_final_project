predictions = predict_labels(X_validation_bag, validation_raw);
% size(predictions)
% size(validationY)
% score = performance_measure(full(predictions), full(validationY))