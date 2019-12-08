# Big Data For Health Team Project

## Introduction

This project is trying to predict ICU readmission in 30 day using [mimic dataset](https://mimic.physionet.org/).

## ETL

We used pyspark for ETL. [src/BD4H_ETL.ipynb](https://github.com/Katvava/BigDataForHealth_TeamProject/blob/master/src/BD4H_ETL.ipynb) is a jupyter notebook running on [google colab](https://colab.research.google.com/notebooks/welcome.ipynb#recent=true). Run the notebook will generate a file

Three major steps for ETL:
**Note** We have uploaded csv data needed by this project to google drive: https://drive.google.com/drive/folders/1MVNHUPmywsONxxakSkW0TO58YYERfKKA?usp=sharing

1. For admission table, get the next unexpected admission and calculate the days until next admission;
2. For notes table, filter discharge summaries;
3. Merge admission table and notes table, generate labels.

This notebook will generate a 'df_adm_notes_clean.pkl', you can find it [here](https://drive.google.com/drive/folders/13PUJeKIsosour6fx4mojJBiPbhIxUovJ?usp=sharing). Move this file to './data' folder for further use.

## Modeling

We tried multiple models:

1. baseline model (random forrest)
2. deep learning model (elmo embedding + GRU)

## Steps of training deep learning model
Input: Please download data for training and testing from https://drive.google.com/drive/folders/1jfWLa2L74cflOJi2tRBdvN4jYGsjU7PZ?usp=sharing

1. Once the data is downloaded, please put them in the following path.

PATH_TRAIN_SEQS = "../data/saved_features/train_data.pkl"
PATH_VALID_SEQS = "../data/saved_features/valid_data.pkl"
PATH_TEST_SEQS = "../data/saved_features/test_data.pkl"

PATH_TRAIN_LEN = "../data/saved_features/train_l_before_padding.pkl"
PATH_VALID_LEN = "../data/saved_features/valid_l_before_padding.pkl"
PATH_TEST_LEN = "../data/saved_features/test_l_before_padding.pkl"

2. To train our model, in terminal, run 
> $python run_train_elmo_gru.py.
This will get pretrained weight saved to "./model_weights/MyRNNELMo.pth'. Learning curves will be saved to './imgs' as well.

3. To test the pretrained model on test data, run
> $python run_test_elmo_gru.py

## Results

Our final results on test set:

Average loss:     0.3939011799170683  
Average accuracy: 87.10595890872963  
Average f1:       0.10909080327602041  
Average auc:      0.5719323360995515  
