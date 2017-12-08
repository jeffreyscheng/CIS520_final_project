predictions = predict_labels_knn(validationX, validation_raw);
% size(predictions)
% size(validationY)

score = performance_measure(full(predictions), full(validationY))