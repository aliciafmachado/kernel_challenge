'''Example code from the Kaggle to read the data and store outputs'''

import numpy as np
import pandas as pd
import src.util.utils as ut
import matplotlib.pyplot as plt

Xtr = np.array(pd.read_csv('../data/Xtr.csv',header=None,sep=',',usecols=range(3072)))
Xte = np.array(pd.read_csv('../data/Xte.csv',header=None,sep=',',usecols=range(3072)))
Ytr = np.array(pd.read_csv('../data/Ytr.csv',sep=',',usecols=[1])).squeeze()

Xtr_gray = ut.rgb_to_grayscale(Xtr)
print(Xtr_gray.shape)

# define your learning algorithm here
# for instance, define an object called ``classifier''
# classifier.train(Ytr,Xtr)


# predict on the test data
# for instance, Yte = classifier.fit(Xte)

# Yte = {'Prediction' : Yte}
# dataframe = pd.DataFrame(Yte)
# dataframe.index += 1
# dataframe.to_csv('Yte_pred.csv',index_label='Id')

print(Xtr.shape,Ytr)
