import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from RDC import RDC_classfication
print('ddddd')
data = pd.read_csv('sample_out.csv')
print(RDC_classfication(data))

