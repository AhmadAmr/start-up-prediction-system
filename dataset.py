import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import  accuracy_score
data = pd.read_csv('sample_out.csv')

Y=data.status
X=data.drop("status",axis='columns')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=17)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
t=accuracy_score(y_test,y_pred)
print(t)

