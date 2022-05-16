# -*- coding: utf-8 -*-  
import pandas as pd
from datasets import load_dataset

def get_dataset():
    data = pd.read_excel("data.xls")
    pd.DataFrame.to_csv(data, "data.csv")
    dataset = load_dataset('csv', data_files="data.csv")
    dataset = dataset['train'].train_test_split(train_size=0.8)
    return dataset
