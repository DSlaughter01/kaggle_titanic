The kaggle_titanic repository shows techniques of data cleaning, feature selection and engineering, appropriate scaling, and the application of machine learning techniques to Kaggle's Titanic dataset, a binary labeled dataset predicting whether people on the titanic will survive. The repository includes the use of scikit-learn and GPy, and will in future include the use of tensorflow.

A number of models are tried out and crossvalidated in order to ascertain the hyperparameters with the best score (based on accuracy, i.e. the number of correct entries out of the total number of entries. 

Solutions are created from main.py and outputted as csv files to the sols direcctory.  

The current best score is 0.77033, as the result of GPy's Gaussian Process implementation

The requirements for running this code can be downloaded using

    pip install -r requirements.txt
