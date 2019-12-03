from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def run_rf_model(train_df, pred_df, label, feat, threshold, accuracy = True): 
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
    
    if accuracy: 
        accuracy = accuracy_score(pred_y, pred)
        return pred_prob, pred, accuracy 
    
    return pred_prob, pred

def run_SVC_model(train_df, pred_df, label, feat, threshold, accuracy = True): 
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
    model = SVC(probability=True)
    
    clf = model.fit(train_X, train_y)
    pred_prob = clf.predict_proba(pred_X)[:,1]
    pred = np.where(pred_prob > threshold, 1, 0)
    
    if accuracy: 
        accuracy = accuracy_score(pred_y, pred)
        return pred_prob, pred, accuracy 
    
    return pred_prob, pred

def run_DT_model(train_df, pred_df, label, feat, threshold, accuracy = True): 
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
    
    if accuracy: 
        accuracy = accuracy_score(pred_y, pred)
        return clf, accuracy 
    
    return clf