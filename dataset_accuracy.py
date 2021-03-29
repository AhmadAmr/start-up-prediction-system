import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from naive import naive_classfication
data = pd.read_csv('sample_out.csv')
print(naive_classfication(data))

