# set up notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import string

def clean_data(adm_path = '', note_path = ''):
    adm_path = '/home/xi/Projects/SS_BigData/Project/data/mimic-iii-clinical-database-1.4/ADMISSIONS.csv'
    note_path = '/home/xi/Projects/SS_BigData/Project/data/mimic-iii-clinical-database-1.4/NOTEEVENTS.csv'

    # read the admissions table
    df_adm = pd.read_csv(adm_path)

    # convert to dates
    df_adm.ADMITTIME = pd.to_datetime(df_adm.ADMITTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_adm.DISCHTIME = pd.to_datetime(df_adm.DISCHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_adm.DEATHTIME = pd.to_datetime(df_adm.DEATHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

    # check to see if there are any missing dates
    print('Number of missing date admissions:', df_adm.ADMITTIME.isnull().sum())
    print('Number of missing date discharges:', df_adm.DISCHTIME.isnull().sum())

    # sort by subject_ID and admission date
    df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])
    df_adm = df_adm.reset_index(drop = True)

    # add the next admission date and type for each subject using groupby
    # you have to use groupby otherwise the dates will be from different subjects
    df_adm['NEXT_ADMITTIME'] = df_adm.groupby('SUBJECT_ID').ADMITTIME.shift(-1)
    # get the next admission type
    df_adm['NEXT_ADMISSION_TYPE'] = df_adm.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)

    # get rows where next admission is elective and replace with naT or nan
    rows = df_adm.NEXT_ADMISSION_TYPE == 'ELECTIVE'
    df_adm.loc[rows,'NEXT_ADMITTIME'] = pd.NaT
    df_adm.loc[rows,'NEXT_ADMISSION_TYPE'] = np.NaN

    # sort by subject_ID and admission date
    # it is safer to sort right before the fill in case something changed the order above
    df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])
    # back fill (this will take a little while)
    df_adm[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']] = df_adm.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']].fillna(method = 'bfill')

    df_adm['DAYS_NEXT_ADMIT']=  (df_adm.NEXT_ADMITTIME - df_adm.DISCHTIME).dt.total_seconds()/(24*60*60)

    # plot a histogram of days between readmissions if they exist
    # this only works for non-null values so you have to filter
    plt.hist(df_adm.loc[~df_adm.DAYS_NEXT_ADMIT.isnull(),'DAYS_NEXT_ADMIT'], bins =range(0,365,30))
    plt.xlim([0,365])
    plt.xlabel('Days between admissions')
    plt.ylabel('Counts')
    plt.show()

    print('Number with a readmission:', (~df_adm.DAYS_NEXT_ADMIT.isnull()).sum())
    print('Total Number:', len(df_adm))

    df_notes = pd.read_csv(note_path)

    print('Number of notes:',len(df_notes))

    df_notes.CATEGORY.unique()

    df_notes.head()

    # look at the first note
    df_notes.TEXT.iloc[0]

    # filter to discharge summary
    df_notes_dis_sum = df_notes.loc[df_notes.CATEGORY == 'Discharge summary']

    df_notes_dis_sum_last = (df_notes_dis_sum.groupby(['SUBJECT_ID', 'HADM_ID']).nth(-1)).reset_index()
    assert df_notes_dis_sum_last.duplicated(['HADM_ID']).sum() == 0, 'Multiple discharge summaries per admission'

    df_adm_notes = pd.merge(df_adm[
                                ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DAYS_NEXT_ADMIT', 'NEXT_ADMITTIME',
                                 'ADMISSION_TYPE', 'DEATHTIME']],
                            df_notes_dis_sum_last[['SUBJECT_ID', 'HADM_ID', 'TEXT']],
                            on=['SUBJECT_ID', 'HADM_ID'],
                            how='left')
    assert len(df_adm) == len(df_adm_notes), 'Number of rows increased'

    print('Fraction of missing notes:', df_adm_notes.TEXT.isnull().sum() / len(df_adm_notes))
    print('Fraction notes with newlines:', df_adm_notes.TEXT.str.contains('\n').sum() / len(df_adm_notes))
    print('Fraction notes with carriage returns:', df_adm_notes.TEXT.str.contains('\r').sum() / len(df_adm_notes))

    df_adm_notes.groupby('ADMISSION_TYPE').apply(lambda g: g.TEXT.isnull().sum())/df_adm_notes.groupby('ADMISSION_TYPE').size()

    df_adm_notes_clean = df_adm_notes.loc[df_adm_notes.ADMISSION_TYPE != 'NEWBORN'].copy()

    print('Fraction of missing notes:', df_adm_notes_clean.TEXT.isnull().sum() / len(df_adm_notes_clean))
    print('Fraction notes with newlines:', df_adm_notes_clean.TEXT.str.contains('\n').sum() / len(df_adm_notes_clean))
    print('Fraction notes with carriage returns:', df_adm_notes_clean.TEXT.str.contains('\r').sum() / len(df_adm_notes_clean))

    df_adm_notes_clean['OUTPUT_LABEL'] = (df_adm_notes_clean.DAYS_NEXT_ADMIT < 30).astype('int')

    print('Number of positive samples:', (df_adm_notes_clean.OUTPUT_LABEL == 1).sum())
    print('Number of negative samples:',  (df_adm_notes_clean.OUTPUT_LABEL == 0).sum())
    print('Total samples:', len(df_adm_notes_clean))

    return df_adm_notes_clean

