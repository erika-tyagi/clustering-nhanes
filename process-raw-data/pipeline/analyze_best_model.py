import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.metrics.ranking import _binary_clf_curve
from functools import reduce
from os import path

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


def feature_importance(model, features): 
    '''
    Returns sorted df with feature names and their importance. 
    '''
    all_vars = list(features)
    features = [v for v in all_vars if v not in (
        'GEOID', 'year_evictions', 'evictions', 'label')]
    feature_importance = pd.DataFrame(zip(features, model.feature_importances_),
                                      columns=['feature', 'importance'])
    feature_importance.sort_values(by='importance', ascending=False, inplace=True)
    return feature_importance


def select_k_blocks(df, k, sort_vars, output_vars): 
    '''
    Returns a df with the top k percent of blocks (based on sort_vars). 
    Only output_vars are returned. 
    '''
    num_blocks = int(df.shape[0]*k)//1 
    selected_blocks = df.nlargest(num_blocks, sort_vars)[output_vars]
    return selected_blocks


def clf_reg_comparison(clf, clf_scores, reg, reg_scores, test_df, k):
    '''
    Returns a df comparing how the clf and reg models perform in predicting 
    the top k percent of blocks in the test_df. Returns the top k percent of 
    blocks in the test_df, along with that block's predicted score and 
    predicted label (from the clf) and predicted evictions (from the reg). 
    Blank predictions indicate that the clf/reg didn't predict that block 
    group to be in the top k percent. 
    '''
    test_blocks = select_k_blocks(test_df, k, ['evictions'], ['GEOID', 'evictions'])
    clf_blocks = select_k_blocks(clf_scores, k, ['score'], ['GEOID', 'score'])
    reg_blocks = select_k_blocks(reg_scores, k, ['pred_evictions'], ['GEOID', 'pred_evictions'])
    merged_df = reduce(lambda left, right: pd.merge(
        left, right, on='GEOID', how='left'), [test_blocks, clf_blocks, reg_blocks])
    merged_df.rename(columns={
        'evictions': 'actual_evictions', 
        'score': 'clf_pred_score', 
        'pred_evictions': 'reg_pred_evictions'}, inplace=True)
    merged_df.sort_values(by='actual_evictions', ascending=False, inplace=True)
    return merged_df


def plot_tree(tree, df, filename, year): 
    '''
    Saves a decision tree. 
    '''
    col_names = list(df.drop(columns=['GEOID', 'year_evictions', 'label']).columns)
    dot_data = StringIO()
    export_graphviz(tree, out_file=dot_data, feature_names=col_names, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png(path.join('results', str(year), filename))


def plot_precision_recall_n(scored_df, vertical_line, title, filename, year):
    '''
    Outputs a precision-recall curve (shows in notebook and saves to filename). 
    '''
    y_true = scored_df.label
    y_score = scored_df.score
    precision_curve, recall_curve, pr_thresholds = _precision_recall_curve_no_truncate(y_true, y_score)
    number_scored = len(y_score)
    pct_above_per_thresh = []
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    margin = 0.01
    ax1.set_ylim([0-margin,1+margin])
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax2.set_ylim([0-margin,1+margin])
    ax2.set_xlim([0-margin,1+margin])
    plt.axvline(x=0.14, color='grey')
    plt.title(title)
    plt.savefig(path.join('results', str(year), filename), dpi=150)
    plt.show()


def _precision_recall_curve_no_truncate(y_true, y_score):
    '''
    Reimplementation of sklearn precision_recall_curve function that doesn't
    truncate thresholds array once recall is 1.
    '''
    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    return precision, recall, thresholds
