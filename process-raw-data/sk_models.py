from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import sklearn

def run_rf_model(train_df, pred_df, label, feat, threshold): 
    '''
    Builds a random forest predictor. 

    Input:  
    - train_df: training dataframe (pandas df)
    - pred_df: prediction dataframe (pandas df)
    - label: label column name (list)
    - feat: feature column names (list)
    - threshold: threshold for converting prediction probability estimates into classifications (float)
    - accuracy: indicator for whether to compute accuracy (boolean)

    Return: 
    - pred_prob: prediction probability estimate for 1 (numpy array)
    - pred: predictions based on thresholds (numpy array)
    - accuracy: prediction accuracy (float)
    '''
    train_X, pred_X = train_df[feat], pred_df[feat]
    train_y = train_df[label].values.ravel()
    pred_y = pred_df[label].values.ravel()
    model = RandomForestClassifier(n_estimators = 10)
    
    clf = model.fit(train_X, train_y)
    pred_prob = clf.predict_proba(pred_X)[:,1]
    pred = np.where(pred_prob > threshold, 1, 0)
    
    accuracy = accuracy_score(pred_y, pred)
    feat_importance = feature_importance(clf, feat)
    
    return accuracy, feat_importance

def run_SVC_model(train_df, pred_df, label, feat, threshold): 
    '''
    Builds a support vector machine model 

    Input:  
    - train_df: training dataframe (pandas df)
    - pred_df: prediction dataframe (pandas df)
    - label: label column name (list)
    - feat: feature column names (list)
    - threshold: threshold for converting prediction probability estimates into classifications (float)
    - accuracy: indicator for whether to compute accuracy (boolean)

    Return: 
    - pred_prob: prediction probability estimate for 1 (numpy array)
    - pred: predictions based on thresholds (numpy array)
    - accuracy: prediction accuracy (float)
    '''    
    train_X, pred_X = train_df[feat], pred_df[feat]
    train_y = train_df[label].values.ravel()
    pred_y = pred_df[label].values.ravel()
    model = SVC(probability=True, kernel = 'linear')
    
    clf = model.fit(train_X, train_y)
    pred_prob = clf.predict_proba(pred_X)[:,1]
    pred = np.where(pred_prob > threshold, 1, 0)

    accuracy = accuracy_score(pred_y, pred)
    feat_importance = feature_importance(clf, feat)
    
    return accuracy, feat_importance

def run_DT_model(train_df, pred_df, label, feat, threshold): 
    '''
    Builds a stump decision tree

    Input:  
    - train_df: training dataframe (pandas df)
    - pred_df: prediction dataframe (pandas df)
    - label: label column name (list)
    - feat: feature column names (list)
    - threshold: threshold for converting prediction probability estimates into classifications (float)
    - accuracy: indicator for whether to compute accuracy (boolean)

    Return: 
    - pred_prob: prediction probability estimate for 1 (numpy array)
    - pred: predictions based on thresholds (numpy array)
    - accuracy: prediction accuracy (float)
    '''   
    train_X, pred_X = train_df[feat], pred_df[feat]
    train_y = train_df[label].values.ravel()
    pred_y = pred_df[label].values.ravel()
    model = DecisionTreeClassifier(max_depth=1)
    
    clf = model.fit(train_X, train_y)
    pred_prob = clf.predict_proba(pred_X)[:,1]
    pred = np.where(pred_prob > threshold, 1, 0)
    
    accuracy = accuracy_score(pred_y, pred)
    feat_importance = feature_importance(clf, feat)
    
    return clf, accuracy, feat_importance

def feature_importance(model, features): 
    '''
    Returns sorted df with feature names and their importance. 
    '''

    if (type(model) == sklearn.linear_model.logistic.LogisticRegression or 
        type(model) == sklearn.svm.SVC):
        coefs = list(model.coef_)[0]
        feature_importance = pd.DataFrame(zip(features, coefs),
                                      columns=['coef', 'value'])
        feature_importance = feature_importance.iloc[(-feature_importance['value'].abs()).argsort()]
        # feature_importance.sort_values(by='value', ascending=False, inplace=True)

    else:
        all_vars = list(features)
        features = [v for v in all_vars if v not in (
            'unique_id', 'label')]
        feature_importance = pd.DataFrame(zip(features, model.feature_importances_),
                                          columns=['feature', 'importance'])
        feature_importance.sort_values(by='importance', ascending=False, inplace=True)
    
    return feature_importance