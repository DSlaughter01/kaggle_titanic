"""
This file is the solution to the Titanic dataset provided by Kaggle.
The problem is an example of binary classification
The goal is to predict which passengers on the Titanic will live, and which will die, using ML.
"""

# Step 0: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GPy

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

###############################################################
""" STEP 1: Import and view the data from train.""" 
train_data = pd.read_csv("train.csv")

print("Training data head")
print(train_data.head(), '\n')

# From info we know that the column Cabin has a lot of null values, as does Age. Embarked has 2.
print("Data info")
print(train_data.info(), '\n')

# Likely outliers in Fare and Age - high maximum (512, 80), low upper quartile in comparison (31, 38)
print(train_data.describe(), '\n')


###############################################################
""" STEP 2: Clean the data (in a function so it can be easily applied to the test data)"""

def clean_data(df) :

    # Remove unwanted observations - check for duplicates (there are none, but 2nd line shows how to remove if there were)
    print("Duplicated values: ", df.duplicated().sum(), '\n')
    data = df.drop_duplicates()

    # Fix structural errors
    # Handle missing data - for the moment in cabin set it to Z, and backfill age
    df["Cabin"] = df["Cabin"].fillna('Z')
    df["Age"] = df["Age"].bfill()
    df["Embarked"] = df["Embarked"].bfill()

    # One Hot Encoding for Sex and Embarked, to change from strings to binary categories (over more columns than the original using dummy variables)
    ohe_sex = pd.get_dummies(df["Sex"], drop_first = True)
    ohe_emb = pd.get_dummies(df["Embarked"], drop_first = True)

    # Change ticket format (how?) - or remove if there are no duplicates:
    tick = df["Ticket"]
    diff = abs(len(set(tick)) - len(list(tick)))
    if diff == 0: print("No duplicate ticket values")
    # else: print(f"{diff} Duplicate ticket values")

    # Extract the letter from the cabin (A to E, and T) - F imposed earlier for passengers with no cabin, and convert to numerical categories
    df["Cabin"] = df["Cabin"].str[0]
    df["Cabin"] = df["Cabin"].astype("category").cat.codes 

    # Manage unwanted outliers
    # Detect with IsolationForest

    # There aren't that many columns compared to the amount of observations, so not too worried about the curse of dimensionality/feature selection
    # Stitch train_data together: remove unwanted columns and add OHE columns
    df = df.drop(columns = ["Name", "Ticket", "PassengerId", "Sex", "Embarked"])
    df = pd.concat([df, ohe_sex, ohe_emb], axis = 1)

    # Change boolean columns to integer
    df["male"] = df["male"].astype(int)
    df["Q"] = df["Q"].astype(int)
    df["S"] = df["S"].astype(int)

    # Final catch for all nan values after conversion of all columns to numeric
    df.fillna(df.mean(), inplace=True)

    return df


###############################################################
""" Step 3: Separate and scale training data """

train_data = clean_data(train_data)
print(train_data.info(), '\n')

# Separate into X and Y
X_train = train_data.values[:, 1:]
Y_train = train_data.values[:, 0]

# Scale and transform ONLY nonbinary categories
scaler = StandardScaler()
bool_cat, non_bool = X_train[:, :6], X_train[:, 6:]
scaled_non_bool = scaler.fit_transform(non_bool)
X_train = np.hstack((scaled_non_bool, bool_cat))

###############################################################
""" Step 4: Train models"""

# Model 1 - SVC using RBF
# Hyperparameters 
c_range = [10 ** i for i in range(-3, 5)]
gamma_range = ["scale", "auto", 0.2, 0.4, 0.6, 0.8, 1.0]
param_var = {"C" : c_range,
             "gamma": gamma_range}

nonlinear_svc = SVC()

# Fit and train with grid search
nl_svc_gs = GridSearchCV(estimator = nonlinear_svc,
                         param_grid = param_var,
                         cv = 5)

nl_svc_gs.fit(X_train, Y_train)
nl_svc_bp = nl_svc_gs.best_params_
nl_svc_train_score = nl_svc_gs.score(X_train, Y_train)

# Show results
nl_svc_res_dict = {"Train accuracy": nl_svc_train_score,
                   "Optimum C": nl_svc_bp["C"],
                   "Optimum gamma": nl_svc_bp["gamma"]}

print(f"SVC: {nl_svc_res_dict}")

# Model 2 - Gaussian Process using the GPy library. 
# GP classification requires label conversion to {-1, 1}
Y_GP_train = (Y_train * 2 - 1).reshape(-1, 1)

# Use an RBF kernel - after looking at the source code (https://gpy.readthedocs.io/en/deploy/_modules/GPy/models/gp_classification.html#GPClassification) if kernel = None (default) then kernel = kern.RBF(X.shape[1]), so no specification needed in my code

# Create a classification model
gp_model = GPy.models.GPClassification(X_train, Y_GP_train)

# Noise is assigned internally through the likelihood function

# Calling optimize iteratively seeks optimal hyperparameter values
gp_model.optimize()

# Predict returns probabilities. Set the threshold at 0.5 to get results of 0 and 1 as required by this problem
gp_train_pred, _ = gp_model.predict(X_train)
gp_train_labels = np.where(gp_train_pred >= 0.5, 1, 0)

# Find the training accuracy (which is the same thing as sklearn's .score() method does for binary classification)
sol = (gp_train_labels.flatten() - Y_train)
correct = 0
for i in sol: 
    if i == 0:
        correct += 1

print(f"Training accuracy using a GP: {correct / len(sol)}")

# Prepare testing data 
test_data = pd.read_csv("test.csv")
test_pass_id = test_data["PassengerId"].values
test_data = clean_data(test_data)
X_test = test_data.values
print(test_data.info(), '\n')

###############################################################
""" Step 5: Predict on test data"""

# Scale and transform ONLY nonbinary categories
scaler = StandardScaler()
bool_cat, non_bool = X_test[:, :6], X_test[:, 6:]
scaled_non_bool = scaler.fit_transform(non_bool)
X_test = np.hstack((scaled_non_bool, bool_cat))

# Predict using SVC - 0.76076
svc_test_pred = nl_svc_gs.predict(X_test)
svc_test_pred = {"PassengerId" : test_pass_id,
        "Survived" : svc_test_pred.astype(int)}

svc_test_pred = pd.DataFrame(svc_test_pred)
svc_test_pred.to_csv("sols/svc_sol.csv", index = False)

# Predict using GP using 0.5 as a threshold
gp_test_pred, _ = gp_model.predict(X_test)
gp_test_pred = np.where(gp_test_pred >= 0.5, 1, 0).flatten()
gp_test_pred = {"PassengerId" : test_pass_id,
        "Survived" : gp_test_pred.astype(int)}

gp_test_pred = pd.DataFrame(gp_test_pred)
gp_test_pred.to_csv("sols/gp_sol.csv", index = False)