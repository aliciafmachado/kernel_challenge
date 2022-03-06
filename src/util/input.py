"""
Util functions to read the data.
"""

import numpy as np
import pandas as pd


def read_data(data_path="data"):
    """
    Read the data from the data_path.
    """
    Xtr = np.array(pd.read_csv('data/Xtr.csv',header=None,sep=',',usecols=range(3072)))
    Xte = np.array(pd.read_csv('data/Xte.csv',header=None,sep=',',usecols=range(3072)))
    Ytr = np.array(pd.read_csv('data/Ytr.csv',sep=',',usecols=[1])).squeeze()
    return Xtr, Ytr, Xte