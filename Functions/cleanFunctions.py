#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:05:19 2018

@author: burnhamd
"""
'''This set of functions provides dataframe cleaning functionality.'''
##Import statements##
import pandas as pd
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import *
import matplotlib.cm as cm
from sklearn.multiclass import OneVsRestClassifier

'''Function to check if an input dataframe contains at least one string 
that is convertable to a float value. Returns boolean value indicating whether
the input string is convertable.  Used in the case that the string values
in the dataframe are not unicode.'''
def is_number(df, column):
    is_float = np.empty(0)
    for index in df.index.tolist():
        try:
            float(df.loc[index][column])
            is_float = np.append(is_float, True)
        except ValueError:
            is_float = np.append(is_float, False)
    if sum(is_float) >= 1:
        return True
    else:
        return False

'''Function checks if a column of data in a dataframe contains binary information.
Takes in column name string, and dataframe object. Returns boolean of whether
the dataframe column is binary.'''
def notBinary(df, column):
        u = df[column].unique().tolist()#grabs unique entries in dataframe column
        has_nan = df[column].isnull().values.any() #checks if column has NaN
        #checks if the column contains binary data, and checks
        #if a column is binary with nan values as well
        notBinary = not (((len(u) == 2) & ((0 in u) & (1 in u))) or 
                         ((len(u) == 3) & ((0 in u) & (1 in u) & has_nan)))
        return notBinary

'''Function that counts the number of null entries in a dataframe, the
number of rows with atleast one null entry, and the number of columns with
atleast one null entry.

Inputs:
    df: dataframe pandas object

Returns:
    count: Total number of missing entries
    null_entries: A numpy array with column one indicating the dataframe row
                    index that has at least one missing value, and column two
                    indicating the number of missing values for that index.
    null_categories: A numpy array with column one indicating the dataframe
                        header that has at least one missing value in its
                        column, and column two indicating the number of missing
                        values for each header.'''
def countMissing(df):
    count = 0
    null_entries = np.empty(0)
    null_categories = np.empty(0)
    for column in df:
        for index in df.index.tolist():
            if((df.loc[index][column] == "?") or (df.loc[index][column] == " ")
            or (df.loc[index][column] == "") or (df.loc[index][column] == "NA")
            or (pd.isna(df.loc[index][column]))):            
                count = count + 1
                null_entries = np.append(null_entries, index)
                null_categories = np.append(null_categories, column)

    #organizes unique indice values that have NaNs, with corresponding counts
    #of NaNs per row.
    ne, ec = np.unique(null_entries, return_counts=True)
    null_entries = np.asarray((ne, ec)).T
    nc, cc = np.unique(null_categories, return_counts=True)
    null_categories = np.asarray((nc, cc)).T
    
    return count, null_entries, null_categories


##Funtion that fills in missing values##
'''Function that fills in missing values.  Takes in a dataframe of any length
and removes missing values marked by ?, blank space, or empty entries.  
Returns the number of missing entries, a numpy array of the indices of rows 
with missing entries and the frequency of missing entries per row, and a numpy 
array of the columns with missing entries and the frequency of missing entries per
column.'''
def replaceMissing(df):
    c, rm, cm = countMissing(df)
    for column in df:
        for index in df.index.tolist():
            #finds locations of missing values marked by ? or blank space
            if((df.loc[index][column] == "?") or (df.loc[index][column] == " ")
            or (df.loc[index][column] == "") or (df.loc[index][column] == "NA")):
                df.at[index, column] = np.nan#replaces missing value with NaN
    
    return c, rm, cm

'''Function that cleans the data by first replacing missing values  
Takes in a dataframe of any length as an input, and modifies the dataframe in place.  
Returns a numpy array of the indices of rows with missing entries, and a numpy array of the column
names with missing entries.

Inputs:
    df: Dataframe pandas object
    

Returns:

    count: Total number of missing entries
    null_entries: A numpy array with column one indicating the dataframe row
                    index that has at least one missing value, and column two
                    indicating the number of missing values for that index.
    null_categories: A numpy array with column one indicating the dataframe
                        header that has at least one missing value in its
                        column, and column two indicating the number of missing
                        values for each header.'''
def clean(df):
    count, rows_missing, columns_missing = replaceMissing(df)

    return count, rows_missing, columns_missing
            
'''Function that plots distributions of each non-binary, numeric dataframe
attribute column.  NaN terms are dropped from the distribution for plotting.'''
def plotDistributions(df):
    for column in df:
        if notBinary(df, column): #binary columns excluded from cleaning operations
            #checks if the column datatype is numeric
            if(df[column].dtype == np.float64 or df[column].dtype == np.int64):
                dist = df.loc[:, column]
                plt.figure()
                plt.title(column)
                plt.hist(dist.dropna())#ignores NaN values
 
'''Function that removes missing values. Returns a new dataframe with NaN values removed using median imputation.'''               
def medianImputation(df):
    df_copy = df.copy()
    clean(df_copy)
    for column in df_copy:
        #checks if column is binary to avoid replacing NaN with median and 
        #guessing a binary state when none is indicated.  Also checks that 
        #the column dtype is numeric and thus has a median value.
        if ((notBinary(df_copy, column)) & 
        ((df_copy[column].dtype == np.float64) or 
         (df_copy[column].dtype == np.int64))):
            
            HasNan = np.isnan(df_copy.loc[:,column])
            df_copy.loc[HasNan, column] = np.nanmedian(df_copy.loc[:, column])
            
    return df_copy

'''Function that normalizes each numeric, non-binary column of the inputted
dataframe.  The function takes in a boolean value to indicate whether to 
use a min max normalization or the standard z-score normalization method. 
Returns a new dataframe with normalized values.'''
def normalizer(df, minMax):
    df_copy = df.copy()
    for h in list(df):
        if notBinary(df, h) & is_number(df, h):
            #creates array of the values of the dataframe column
            x = df[[h]].values.astype(float)
            # Create a minimum and maximum processor object
            if minMax:
                minmax_scale = preprocessing.MinMaxScaler()
                x_scaled = minmax_scale.fit_transform(x)
            else:
                z_scaler = preprocessing.StandardScaler()
                # Create an object to transform the data to fit minmax processor
                x_scaled = z_scaler.fit_transform(x)
            df_copy[h] = x_scaled
    return df_copy

# split a dataset in dataframe format, using a given ratio for the testing set
def splitter(df, ratio, seed, pred_col):
    #index object with indices of overall dataframe
    all_ind = pd.Index(df.index.values)
    
    if ratio >= 1: 
        print ("Parameter r needs to be smaller than 1!")
        return
    elif ratio <= 0:
        print ("Parameter r needs to be larger than 0!")
        return
    #random split of the overall dataframe into a test subset
    df_test = df.sample(frac=ratio, random_state=seed)
    #list of the indices of the test dataframe split
    ind_test = df_test.index.values.tolist()
    #boolean array of where test indices are in the overall dataframe
    split_indices = all_ind.isin(ind_test)
    df_train = df[~split_indices] #dataframe containing data to train on
    df_train_out = df_train[pred_col] #output to predict in training dataset
    df_test_out = df_test[pred_col] #output to predict in testing dataset
    #remove output column from training dataset
    df_train = df_train.drop(pred_col, axis = 1)
    #remove output column from testing dataset
    df_test = df_test.drop(pred_col, axis = 1)
    
    #return the testing and training dataset with respective outputs to predict
    return df_test, df_test_out, df_train, df_train_out


def charactester(actualValues, predictedValues):
    aScore = accuracy_score(actualValues, predictedValues)
    # Confusion Matrix
    CM = confusion_matrix(dactualValues, predictedValues)
    class_num = len(actualValues.unique())
    if class_num > 2:
        P = precision_score(actualValues, predictedValues, average='weighted')
        R = recall_score(actualValues, predictedValues, average='weighted')
        F1 = f1_score(actualValues, predictedValues, average='weighted')
        print("Cannot perform ROC on multiclass")
    else:
        P = precision_score(actualValues, predictedValues)
        R = recall_score(actualValues, predictedValues)
        F1 = f1_score(actualValues, predictedValues)
    
        # ROC analysis
        LW = 1.5 # line width for plots
        LL = "lower right" # legend location
        LC = 'darkgreen' # Line Color
        
        # False Positive Rate, True Posisive Rate, probability thresholds
        fpr, tpr, th = roc_curve(actualValues, predictedValues)
        AUC = auc(fpr, tpr)
        a = {'Results': [aScore, P, R, F1, fpr, tpr, AUC, th]}
        rowNames = ['Accuracy', 'Precision', 'Recall', 'F1', 
                    'True Positive Rate', 'False Positive Rate',
                    'AUC', 'Probability Thresholds']
        analyses = pd.DataFrame(a, index = rowNames)
        
        #ROC Plotting
        plt.figure()
        plt.title('Receiver Operating Characteristic curve example')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FALSE Positive Rate')
        plt.ylabel('TRUE Positive Rate')
        plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
        # reference line for random classifier
        plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--')
        plt.legend(loc=LL)
        plt.show()

'''Function that plots an ROC curve per class of a multiclass classifier case
by using the One Vs Rest technique.  This function was adapted from code
at: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
demonstrating the multiclass implementation of ROC curves for classifier
evaluation.

Inputs:
    X_train: dataframe object containing the data to train on.
    y_train: dataframe object containing the label binarized class outputs of 
            the training set.
    X_test: dataframe object containing the data for testing the model
    y_test: dataframe object containing the label binarized class outputs of
            the test set.
    model: classifier object delineating the model to fit
    classes: list containing the classes of the data to be used in plotting

Outputs:
    ROC curve plot containing curves for each class, as well as the micro and
    macro average ROC curves.
'''
def multiclassROC(X_train, y_train, X_test, y_test, model, classes):
    classifier = OneVsRestClassifier(model)
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    n_classes = y_train.shape[1]
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    lw=2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-av ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-av ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    start = 0.0
    stop = 1.0
    number_of_lines= n_classes
    cm_subsection = np.linspace(start, stop, number_of_lines) 
    colors = [ cm.jet(x) for x in cm_subsection ]
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='Class {0} (area = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve Multiclass')
    plt.legend(bbox_to_anchor=(1.00, 1.00))
    plt.show()
    
    
"""
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`.
Adapted from code at: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    
    
    