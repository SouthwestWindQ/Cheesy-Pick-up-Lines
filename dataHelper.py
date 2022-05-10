# -*- coding: utf-8 -*-  
import pandas as pd
from datasets import load_dataset

def get_dataset():
    data = pd.read_excel("trial.xls")
    pd.DataFrame.to_csv(data, "trial.csv")
    dataset = load_dataset('csv', data_files="trial.csv")
    dataset = dataset['train'].train_test_split(train_size=0.8)
    return dataset
