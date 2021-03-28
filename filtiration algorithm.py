import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
data = pd.read_csv('sample_out.csv')
Cat = input('Enter the category name')
country_code = int(input('Enter the country code'))
filt2 = (data[Cat] == 1)
filt1 = (data['country_code'] == country_code)
country = data.loc[filt1]
final = country.loc[filt2]
if final.empty or len(final.index) < 10 :
    print('No statrtups from this category enough to predict')
else:
    Y = final.status
    X = final.drop("status", axis='columns')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=17)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    t = accuracy_score(y_test, y_pred)
    print(t*100, '%')
    print(final['fundintotal_usd'].mean())





