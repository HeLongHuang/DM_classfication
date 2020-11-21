import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 100000000)
pd.set_option('display.width', 100000000)
pd.set_option('display.max_colwidth', 10000000)

class process_data():
    def __init__(self):
        self.train_data_path = 'data/raw/kddtrain2020.txt'
        self.test_data_path = 'data/raw/kddtest2020.txt'
        self.train_data_csv_savepath = 'data/processed/train.csv'
        self.test_data_csv_savepath = 'data/processed/test.csv'
    def get_train_data_csv(self):
        data = np.loadtxt(self.train_data_path, dtype=str)
        columns = []
        for i in range(100):
            columns.append("attribute" + str(i))
        columns.append("label")
        data = pd.DataFrame(data, columns=columns,dtype='float32')
        data.to_csv(self.train_data_csv_savepath)
        return data
    def get_test_data_csv(self):
        columns = []
        for i in range(100):
            columns.append("attribute" + str(i))
        data = pd.read_table(self.test_data_path, names=columns,dtype='float32')
        return data







