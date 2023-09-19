import pandas as pd
import numpy as np
import math
from decimal import Decimal
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle

def preprocess_data(df):
    # Preprocess the data
    df = df.dropna()
    return df

def select_features(df):
    # Select the relevant features
    features = ['AGE', 'GENDER', 'T3', 'T4U', 'TSH']
    X = df[features]
    y = df['DIAGNOSIS']
    return X, y

def max_times(l):
    ans=""
    for i in l:
        if l.count(i)>2:
            ans=i
            break
    return ans



def train_model(X, y, model_name):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Choose the appropriate model
    if model_name == 'Decision Tree':
        clf = DecisionTreeClassifier()
    elif model_name == 'Logistic Regression':
        clf = LogisticRegression(max_iter=6000)
    elif model_name == 'Random Forest':
        clf = RandomForestClassifier()
    elif model_name == 'Naive Bayes':
        clf = GaussianNB()
    else:
        raise ValueError(
            "Invalid model name. Choose from 'Decision Tree', 'Logistic Regression', 'Random Forest','Naive Bayes'")

    # Train the model
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,clf

def make_prediction(clf, X_new):
    # Make a prediction for a single person
    y_new = clf.predict(X_new)
    return y_new

def find_accuracy(Pregnancy_status):
    Decision_tree = []
    Logistic_regression = []
    Random_forest = []
    Naive_bayes = []


    if Pregnancy_status:
        dataset = 'Dataset\dataset0387_pregnant.csv'
        Decision_tree_model = "Trained_models_for_pregnant/trained_model_by_Decision_tree_for_pregnant.pkl"
        Logistic_regression_model = "Trained_models_for_pregnant/trained_model_by_Logistic_regression_for_pregnant.pkl"
        Random_forest_model = "Trained_models_for_pregnant/trained_model_by_Random_forest_for_pregnant.pkl"
        Naive_bayes_model = "Trained_models_for_pregnant/trained_model_by_Naive_bayes_for_pregnant.pkl"


    else:
        dataset = 'Dataset\dataset0387_normal.csv'
        Decision_tree_model = "Trained_models_for_normal/trained_model_by_Decision_tree_for_normal.pkl"
        Logistic_regression_model = "Trained_models_for_normal/trained_model_by_Logistic_regression_for_normal.pkl"
        Random_forest_model = "Trained_models_for_normal/trained_model_by_Random_forest_for_normal.pkl"
        Naive_bayes_model = "Trained_models_for_normal/trained_model_by_Naive_bayes_for_normal.pkl"


    df = pd.read_csv(dataset)
    # Preprocess the data
    df = preprocess_data(df)

    # Select the relevant features
    X, y = select_features(df)

    accuracy,clf = train_model(X, y, 'Decision Tree')
    Decision_tree.append(accuracy)
    with open(Decision_tree_model, 'wb') as f:
        pickle.dump(clf, f)

    accuracy,clf = train_model(X, y, 'Random Forest')
    Random_forest.append(accuracy)
    with open(Random_forest_model, 'wb') as f:
        pickle.dump(clf, f)

    accuracy,clf = train_model(X, y, 'Logistic Regression')
    Logistic_regression.append(accuracy)
    with open(Logistic_regression_model, 'wb') as f:
        pickle.dump(clf, f)

    accuracy,clf = train_model(X, y, 'Naive Bayes')
    Naive_bayes.append(accuracy)
    with open(Naive_bayes_model, 'wb') as f:
        pickle.dump(clf, f)

    for i in range(0,4):
        accuracy, clf = train_model(X, y, 'Decision Tree')
        Decision_tree.append(accuracy)
        accuracy, clf = train_model(X, y, 'Random Forest')
        Random_forest.append(accuracy)
        accuracy, clf = train_model(X, y, 'Logistic Regression')
        Logistic_regression.append(accuracy)
        accuracy, clf = train_model(X, y, 'Naive Bayes')
        Naive_bayes.append(accuracy)

    Dec_Avg=round((sum(Decision_tree)/len(Decision_tree))*100,2)
    Dec_max=round((max(Decision_tree)*100),2)

    Ran_Avg=round((sum(Random_forest)/len(Random_forest))*100,2)
    Ran_max=round((max(Random_forest) * 100),2)

    Log_Avg=round((sum(Logistic_regression)/len(Logistic_regression))*100,2)
    Log_max=round((max(Logistic_regression) * 100),2)

    Naive_Avg=round((sum(Naive_bayes)/len(Naive_bayes))*100,2)
    Naive_max=round((max(Naive_bayes) * 100),2)

    return Dec_Avg,Dec_max,Ran_Avg,Ran_max,Log_Avg,Log_max,Naive_Avg,Naive_max






def predict(model_name,Pregnancy_status,X_new):

    if Pregnancy_status:
        dataset = 'Dataset\dataset0387_pregnant.csv'
        Decision_tree_model = "Trained_models_for_pregnant/trained_model_by_Decision_tree_for_pregnant.pkl"
        Logistic_regression_model = "Trained_models_for_pregnant/trained_model_by_Logistic_regression_for_pregnant.pkl"
        Random_forest_model = "Trained_models_for_pregnant/trained_model_by_Random_forest_for_pregnant.pkl"
        Naive_bayes_model = "Trained_models_for_pregnant/trained_model_by_Naive_bayes_for_pregnant.pkl"


    else:
        dataset = 'Dataset\dataset0387_normal.csv'
        Decision_tree_model = "Trained_models_for_normal/trained_model_by_Decision_tree_for_normal.pkl"
        Logistic_regression_model = "Trained_models_for_normal/trained_model_by_Logistic_regression_for_normal.pkl"
        Random_forest_model = "Trained_models_for_normal/trained_model_by_Random_forest_for_normal.pkl"
        Naive_bayes_model = "Trained_models_for_normal/trained_model_by_Naive_bayes_for_normal.pkl"

    df = pd.read_csv(dataset)
    df = preprocess_data(df)
    X, y = select_features(df)


    if model_name == 'Decision Tree':
        with open(Decision_tree_model, 'rb') as f:
            clf = pickle.load(f)
        X_new = pd.DataFrame(X_new, columns=X.columns)
        pred = make_prediction(clf, X_new)


    elif model_name == 'Logistic Regression':
        with open(Logistic_regression_model, 'rb') as f:
            clf = pickle.load(f)
        X_new = pd.DataFrame(X_new, columns=X.columns)
        pred = make_prediction(clf, X_new)


    elif model_name == 'Random Forest':
        with open(Random_forest_model, 'rb') as f:
            clf = pickle.load(f)
        X_new = pd.DataFrame(X_new, columns=X.columns)
        pred = make_prediction(clf, X_new)


    elif model_name == 'Naive Bayes':
        with open(Naive_bayes_model, 'rb') as f:
            clf = pickle.load(f)
        X_new = pd.DataFrame(X_new, columns=X.columns)
        pred = make_prediction(clf, X_new)


    else:
        raise ValueError(
            "Invalid model name. Choose from 'Decision Tree', 'Logistic Regression', 'Random Forest','Naive Bayes'")

    return pred

def final_prediction(Pregnancy_status,X_new):
    i=0
    final=[]

    if Pregnancy_status:
        while i<5:
            Ran=predict("Random Forest",1,X_new)
            a=find_accuracy(1)
            print(Ran)
            if (Ran):
                final.append(Ran)
                i=i+1

    else:
        while i<5:
            Ran = predict("Random Forest", 0, X_new)
            Dec = predict("Decision Tree",0,X_new)
            print(Ran,Dec)
            if (Ran==Dec):
                final.append(Ran)
                i=i+1


    ans=max_times(final)
    return ans








