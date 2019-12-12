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

import pandas as pd
import numpy as np
from sklearn import preprocessing

# Master function to clean a split
def clean_split(split, components=False):
    
    if components:
        train_df, test_df = split
        train_df = clean_overall_data(train_df, components=True)
        test_df = clean_overall_data(test_df, components=True)

        features_generator = get_feature_generators(train_df, components=True)
        train_df, test_df = \
        clean_and_create_features(train_df, test_df, features_generator, components=True)

    else:
        train_df, test_df = split
        train_df = clean_overall_data(train_df)
        test_df = clean_overall_data(test_df)

        features_generator = get_feature_generators(train_df)
        train_df, test_df = \
        clean_and_create_features(train_df, test_df, features_generator)

    return train_df, test_df


####### FUNCTIONS TO RUN PRE-TRAIN_TEST SPLIT ######

def clean_overall_data(complete_df, components=False):
    '''
    Execute cleaning steps prior to splitting into test and train
    '''
    if components:
        complete_df = convert_to_numeric(complete_df, ['RIDAGEYR', 'INDFMPIR','PC1','PC2','PC3',
            'PC4', 'PC5', 'PC6', 'PC7','BMXBMI','BPQ020'])

    else:
        complete_df = convert_to_numeric(complete_df, ['RIDAGEYR', 'INDFMPIR','assignment_kmeans',
            'BMXBMI','BPQ020'])
        complete_df['cluster_1'] = np.where(complete_df['assignment_kmeans'] == 1, 1, 0)
    
    #Create the label columns
    complete_df['has_had_hbp'] = np.where(complete_df['BPQ020'] == 1, 1, 0)
    complete_df['is_obese'] = np.where(complete_df['BMXBMI'] >= 30.0, 1, 0)
    
    return complete_df

####### FUNCTIONS TO RUN TO FIT ON TRAIN DATA ONLY ######

def get_feature_generators(train_df, components=False):
    
    feature_generator_dict = {}

    if components:
        feature_generator_dict['scalers'] = create_scaler(train_df,['RIDAGEYR', 'INDFMPIR','PC1','PC2','PC3',
            'PC4', 'PC5', 'PC6', 'PC7'])

    else:
        feature_generator_dict['scalers'] = create_scaler(train_df,['RIDAGEYR', 'INDFMPIR'])    

    return feature_generator_dict


####### FUNCTIONS TO RUN ON BOTH TRAIN AND TEST SEPARATELY ######

def clean_and_create_features(train_df, test_df, feature_generator_dict=None, components=False):

    #Mark rows where imputation is going to occur
    train_df = create_impute_flag(train_df)
    test_df = create_impute_flag(test_df)

    #Impute missing values

    train_df = train_df.fillna(train_df.median())
    train_df = train_df.dropna(axis=1, how='all')
    test_df = test_df.fillna(test_df.median())
    test_df = test_df.dropna(axis=1, how='all')

    test_df = check_col_match(train_df, test_df)

    if feature_generator_dict:
        scalers = feature_generator_dict['scalers']
        #binaries = feature_generator_dict['binaries']

        train_df, test_df = scale_data(train_df, test_df, scalers)
        #train_df, test_df = binarize_data(train_df, test_df, binaries)

    train_df = pd.get_dummies(data=train_df, columns=['RIAGENDR','RIDRETH1'])
    test_df = pd.get_dummies(data=test_df, columns=['RIAGENDR','RIDRETH1'])

    cols_to_drop = {'year','RIDAGEYR', 'INDFMPIR','PC1','PC2','PC3', 
    'PC4', 'PC5', 'PC6', 'PC7','assignment_kmeans','BMXBMI','BPQ020','SEQN'}

    #cleaning out columns where there are no differentiations - specifically where we
    #don't impute at all here
    train_df = train_df.loc[:, (train_df != 0).any(axis=0)]
    test_df = test_df.loc[:, (test_df != 0).any(axis=0)]

    train_df = drop_unwanted_columns(train_df, set(train_df.columns) & cols_to_drop)
    test_df = drop_unwanted_columns(test_df, set(test_df.columns) & cols_to_drop)

    return train_df, test_df


###### HELPER FUNCTIONS ######

def clean_period_as_missing(df):
    '''
    Reformats missing values represented as a period to be a numpy NaN value
    '''
    df = df.replace('^\.$', np.nan, regex=True)
    return df

def clean_missing_vals(df):
    '''
    Reformats known missing number values as nans
    '''
    df = df.replace(-666666666, np.nan)
    return df

def convert_to_numeric(df, cols):
    '''
    Converts columns which contain numbers in string format to integers
    '''
    df[cols] = df[cols].apply(pd.to_numeric)
    return df

def impute_as_zero(df, cols):
    '''
    Imputs missing values as 0 (for where no data means that this field was
    not present, e.g. no crimes occurred)
    '''
    for col in cols:
        df[col].fillna(0.0, inplace=True)

    return df

def scale_data(train_df, test_df, scaler_dict):
    '''
    Scale data on training and apply to testing set leveraging scikitlearn function
    Inputs: training df, testing df, colum
    '''
    for col, scaler in scaler_dict.items():

        if col not in train_df.columns:
            continue

        train_df[col+'_scaled'] = scaler.transform(train_df[[col]])
        test_df[col+'_scaled'] = scaler.transform(test_df[[col]])

    return train_df, test_df

def create_scaler(train_df, cols):
    '''
    Creates scalers for relevant columns in a dataframe
    '''

    scaler_dict = {}

    for col in cols:
        scaler = preprocessing.StandardScaler()
        scaler = scaler.fit(train_df[[col]])
        scaler_dict[col] = scaler

    return scaler_dict

def get_binary_cutoffs(train_df, col_quantile_dict):
    '''
    Create dict of cols and median vals in training data to apply on train + test in
    creating binary columns
    '''
    return_dict = {}

    for col, q in col_quantile_dict.items():
        return_dict[col] = train_df[col].quantile(q=q)

    return return_dict

def binarize_data(train_df, test_df, quantile_dict):
    '''
    Takes median dictionaries
    '''
    for col, quantile in quantile_dict.items():

        if col not in train_df.columns:
            continue
            
        train_df[col+'_binary'] = np.where(train_df[col] >= quantile, 1, 0)
        test_df[col+'_binary'] = np.where(test_df[col] >= quantile, 1, 0)

    return train_df, test_df

def drop_unwanted_columns(df, cols_to_drop):
    '''
    Drops columns that will not be used in feature generation
    '''
    df = df.drop(cols_to_drop, axis=1)
    return df

def get_change_in_feature(df, col, historical_cols):
    '''
    Creates a new feature of % change from t - time horizon year
    '''
    for historical_col in historical_cols:
        time_horizon = historical_col[-1]
        df[col+'_change'+'_'+time_horizon] = (df[col] - df[historical_col])/df[historical_col]
        
        df[col+'_change'+'_'+time_horizon] = np.where((df[col] == 0.0) & (df[historical_col] == 0.0), 
            0.0, df[col+'_change'+'_'+time_horizon])

        df[col+'_change'+'_'+time_horizon] = np.where(df[historical_col] == 0.0, 
            1.0 * df[historical_col], df[col+'_change'+'_'+time_horizon])
    
    return df

def get_pct_feature(df, numerator_col, denom_col):
    '''
    Create a percentage column
    '''
    df[numerator_col+'_percent'] = df[numerator_col]/df[denom_col]

    df[numerator_col+'_percent'] = np.where((df[numerator_col] == 0.0) & (df[denom_col] == 0.0), 
        0.0, df[numerator_col+'_percent'])

    df[numerator_col+'_percent'] = np.where(df[denom_col] == 0.0, 
        np.nan, df[numerator_col+'_percent'])

    return df
def check_col_match(train_df, test_df):
    '''
    Remove cols from test that do not appear in training
    '''
    extra_cols = []

    for column in test_df.columns:
        if column not in train_df.columns:
            extra_cols.append(column)

    test_df = test_df.drop(extra_cols,axis=1)

    return test_df

def create_impute_flag(df):
    for col in df.columns:
        df[col+'_impute_flag'] = np.where(df[col].isna(), 1, 0)
    return df
