from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

def naive_classfication(data):
    Y = data.status
    X = data.drop("status",axis='columns')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=17)
    clf = RandomForestClassifier(n_estimators=1)
    clf = clf.fit(X_train, y_train)
    accuracy = clf.score(X_train, y_train)
    ypredict = clf.predict(X_train)
    print('\nTraining classification report\n', classification_report(y_train, ypredict))
    print( "\n Confusion matrix of training \n", confusion_matrix(y_train, ypredict))
    ypredict = clf.predict(X_test)
    print( '\nTraining classification report\n', classification_report(y_test, ypredict))
    print("\n Confusion matrix of training \n", confusion_matrix(y_test, ypredict))

    return 0




