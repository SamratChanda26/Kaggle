import numpy as np
import pandas as pd

iris = pd.read_csv("IRIS.csv")
print("\n", iris, "\n")

iris.insert(0, 'Id', range(1, 1 + len(iris)))
iris.info()

print("\n", iris.isnull().sum())

print("\n", iris.shape)
print("\n", iris.dtypes)

print("\n", iris.columns)

target = iris['species']
print("\n", target)

df = iris.drop("species", axis=1)
print("\n", df)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.3)

print("\n", x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

X = df
y = target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=0)
print("\n", np.shape(X))
print(np.shape(X_test))
print(np.shape(X_train))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#SVM

from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1, random_state=0)
svm.fit(X_train_std, y_train)
y_pred=svm.predict(X_test_std)
print('\n misclassified samples: %d'%(y_test!=y_pred).sum())
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f'%accuracy_score(y_test, y_pred))

#KNN

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn1 = KNeighborsClassifier(n_neighbors=4)
knn1.fit(X_train_std, y_train)
y_pred = knn1.predict(X_test_std)
print('\n misclassified samples: %d'%(y_test!=y_pred).sum())
print('Accuracy: %.2f'%accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
model = LogisticRegression(solver='liblinear', random_state=2)
model.fit(X_train, y_train)
model = LogisticRegression(solver='liblinear', random_state=0).fit(X, y)
model.predict_proba(X)
model.predict(X)
model.score(X, y)
print("\n", classification_report(y, model.predict(X)))
print(model.score(X, y))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(df, target, test_size=0.2, random_state=0)

param_grid = {'n_estimators' : [50, 100, 150], 'max_depth' : [None, 5, 10], 'min_samples_split' : [2, 5, 10]}

rfc = RandomForestClassifier()

grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv = 5)
grid_search.fit(X_train, y_train)

print("\n Best hyperparameters: ", grid_search.best_params_)
y_pred = grid_search.predict(X_val)
print("Validation accuracy score: ", accuracy_score(y_val, y_pred))

predictions = model.predict(X_test)
print("\n", predictions)

output = pd.DataFrame({'Id' : X.Id, 'Species' : model.predict(X)})
output.to_csv('submission.csv', index = False)
print("Your submission was successfully saved!")