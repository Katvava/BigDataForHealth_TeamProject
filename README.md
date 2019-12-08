# Big Data For Health Team Project

## Introduction

This project is trying to predict ICU readmission in 30 day using [mimic dataset](https://mimic.physionet.org/).

## ETL

We used pyspark for ETL. [src/BD4H_ETL.ipynb](https://github.com/Katvava/BigDataForHealth_TeamProject/blob/master/src/BD4H_ETL.ipynb) is a jupyter notebook running on [google colab](https://colab.research.google.com/notebooks/welcome.ipynb#recent=true).

Three major steps for ETL:
1. For admission table, get the next unexpected admission and calculate the days until next admission;
2. For notes table, filter discharge summaries;
3. Merge admission table and notes table, generate labels.

**Note**: You need upload the dataset to google drive and modify the corresponding path in order to run the notebook.

## Modeling

We tried multiple models:

1. baseline model (random forrest)
2. deep learning model (elmo embedding + GRU)

## Results

Our final results on test set:
Average loss:     0.3939011799170683
Average accuracy: 87.10595890872963
Average f1:       0.10909080327602041
Average auc:      0.5719323360995515
