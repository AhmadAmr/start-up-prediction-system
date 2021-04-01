import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from RDC import RDC_classfication
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
data = pd.read_csv('sample_out.csv')
Cat = input('Enter the category name')
country_code = int(input('Enter the country code'))
filt2 = (data[Cat] == 1)
filt1 = (data['country_code'] == country_code)
country = data.loc[filt1]
final = country.loc[filt2]
fail = data.loc[filt2]
if final.empty or len(final.index) < 10 :
    print('This category not popular in this country, here is the success percent of this category in the world')
    print(RDC_classfication(fail))
    print(final['fundintotal_usd'].mean())
else:
    Y = final.status
    X = final.drop("status", axis='columns')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=17)
    clf = RandomForestClassifier(n_estimators=1)
    clf = clf.fit(X_train, y_train)
    accuracy = clf.score(X_train, y_train)
    ypredict = clf.predict(X_train)
    print('\nTraining classification report\n', classification_report(y_train, ypredict))
    print("\n Confusion matrix of training \n", confusion_matrix(y_train, ypredict))
    ypredict = clf.predict(X_test)
    print('\nTraining classification report\n', classification_report(y_test, ypredict))
    print("\n Confusion matrix of training \n", confusion_matrix(y_test, ypredict))
    print(accuracy)
    print(final['fundintotal_usd'].mean())





