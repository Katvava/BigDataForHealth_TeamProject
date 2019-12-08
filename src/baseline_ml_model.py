# set up notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from get_bag_of_words_feature import get_bow_feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_curve, accuracy_score, auc
from python_get_adm_notes import clean_data
import string

def create_training_testing_data(df_adm_notes_clean):
    # shuffle the samples
    df_adm_notes_clean = df_adm_notes_clean.sample(n=len(df_adm_notes_clean), random_state=42)
    df_adm_notes_clean = df_adm_notes_clean.reset_index(drop=True)

    # Save 30% of the data as validation and test data
    df_valid_test = df_adm_notes_clean.sample(frac=0.60, random_state=42)

    df_test = df_valid_test.sample(frac=0.5, random_state=42)
    df_valid = df_valid_test.drop(df_test.index)

    # use the rest of the data as training data
    df_train_all = df_adm_notes_clean.drop(df_valid_test.index)

    print('Test prevalence(n = %d):' % len(df_test), df_test.OUTPUT_LABEL.sum() / len(df_test))
    print('Valid prevalence(n = %d):' % len(df_valid), df_valid.OUTPUT_LABEL.sum() / len(df_valid))
    print('Train all prevalence(n = %d):' % len(df_train_all), df_train_all.OUTPUT_LABEL.sum() / len(df_train_all))
    print('all samples (n = %d)' % len(df_adm_notes_clean))

    # split the training data into positive and negative
    rows_pos = df_train_all.OUTPUT_LABEL == 1
    df_train_pos = df_train_all.loc[rows_pos]
    df_train_neg = df_train_all.loc[~rows_pos]

    # merge the balanced data
    df_train = pd.concat([df_train_pos.sample(n=15000, random_state=42, replace = True), df_train_neg.sample(n=15000, random_state = 42, replace = True)], axis=0)

    # shuffle the order of training samples
    df_train = df_train.sample(n=len(df_train), random_state=42).reset_index(drop=True)

    print('Train prevalence (n = %d):' % len(df_train), df_train.OUTPUT_LABEL.sum() / len(df_train))

    return df_train, df_test, df_valid

def clean_text(df):
    # This function preprocesses the text by filling not a number and replacing new lines ('\n') and carriage returns ('\r')
    df.TEXT = df.TEXT.fillna(' ')
    df.TEXT =df.TEXT.str.replace('\n',' ')
    df.TEXT =df.TEXT.str.replace('\r',' ')
    return df

def main():
    ETL_generated_notes_path = '../data/df_adm_notes_clean.pkl'

    if not os.path.exists(ETL_generated_notes_path)
        df_adm_notes_clean = clean_data()
    else:
        df_adm_notes_clean = pd.read_pickle(ETL_generated_notes_path)

    df_train, df_test, df_valid = create_training_testing_data(df_adm_notes_clean)

    # preprocess the text to deal with known issues
    df_train = clean_text(df_train)
    df_valid = clean_text(df_valid)
    df_test = clean_text(df_test)

    X_train_tf, y_train, X_valid_tf, y_valid = get_bow_feature(df_train, df_valid)

    zx  = 1

    clf = RandomForestClassifier(max_depth=20, n_estimators=50, max_features=10)
    clf.fit(X_train_tf, y_train)

    y_valid_pred = clf.predict(X_valid_tf)
    y_valid_proba = clf.predict_proba(X_valid_tf)

    accuracy = accuracy_score(y_valid, y_valid_pred)
    f1 = f1_score(y_valid, y_valid_pred)
    fpr, tpr, threshold = roc_curve(y_valid, y_valid_proba[:, 1])
    auc_value = auc(fpr, tpr)

    print('Accuracy: {}'.format(accuracy))
    print('F1: {}'.format(f1))
    print('AUC: {}'.format(auc_value))

if __name__ == '__main__':
    main()

# Accuracy: 0.5860179992174254
# F1: 0.13254987701557802
# AUC: 0.59576684653522