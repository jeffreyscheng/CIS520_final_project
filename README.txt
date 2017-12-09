The following is a description of the files:                                                                     

Below is a description of the models. For each model, I describe what files it is contained in
and how to run our testscript.
Models:
1) Naive Bayes:
    - This is based in predict_labels_nb.m.
    - To call this function, call predict_labels_nb(X_test_bag, test_raw) where
        > X_test_bag = the test dataset.
        > test_raw = the raw tweet data.
    - We load in the training data and train using a naive bayes model.
    - Y_hat (the output) is the predicted labels of the test data passed in
    
    - An example of this is within the scripts/test_script_nb.m script. To run this, simply just run it.
    - This script randomizes the train set to come up with a test set and a train set,
    - It then trains the model and predicts for the test set and prints the score

2) Logistic Regression:
    - This is based in predict_labels.m.
    - To call this function, call predict_labels(X_test_bag, test_raw) where
        > X_test_bag = the test dataset.
        > test_raw = the raw tweet data.
    - We load in the training data and train using a logistic model with cost optimization.
    - Y_hat (the output) is the predicted labels of the test data passed in

    - An example of this is within the scripts/test_script.m script. To run this, simply just run it.
    - This script randomizes the train set to come up with a test set and a train set,
    - It then trains the model and predicts for the test set and prints the score

3) K-Nearest Neighbors:
    - This is based in predict_labels_knn.m.
    - To call this function, call predict_labels_knn(X_test_bag, test_raw) where
        > X_test_bag = the test dataset.
        > test_raw = the raw tweet data.
    - We load in the training data and train using a K-Nearest Neighbors model with K = 15.
    - Y_hat (the output) is the predicted labels of the test data passed in

    - An example of this is within the scripts/test_script_knn.m script. To run this, simply just run it.
    - This script randomizes the train set to come up with a test set and a train set,
    - It then trains the model and predicts for the test set and prints the score

File Description:
1) predict_labels.m
	- MATLAB function template giving the format for code submitted to the leaderboard
	  We will call this function on our end to test your algorithms.

2) perfomance_measure.m
	- MATLAB function to calculate the average cost of your prediction based on the cost matrix.


3) vocabulary.mat
	- Contains a 1x10000 vector of strings containing the 10000 words used to generate the bag-of-words features from the raw text. For a tweet, the bag-of-words features are the counts of appearences in the tweet, of words from the vocabulary.

4) train.mat
	- Contains a 18092x1 struct containing the tweets in raw text
	- Contains a 18092x10000 matrix, where the rows correspond to the tweets and columns correspond to the bag-of-words features for the tweets
	- Contains a 18092x1 vector of labels (joy (1), sadness (2), surprise (3), anger (4) or fear (5))

5) validation.mat
	- Contains a 9098x1 struct containing the tweets in raw text
	- Contains a 9098x10000 matrix containing the bag-of-words features for the tweets, where the rows correspond to the tweets and columns correspond to the bag-of-words features for the tweets
