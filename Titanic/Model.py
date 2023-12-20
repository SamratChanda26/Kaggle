import numpy as np
import pandas as pd

train_data = pd.read_csv("train.csv")
print("\n", train_data.head())

test_data = pd.read_csv("test.csv")
print("\n", test_data.head())

women = train_data.loc[train_data.Sex == 'female'] ["Survived"]
rate_women = sum(women)/len(women)

print("\n% of women who survived: ", rate_women, "\n")

men = train_data.loc[train_data.Sex == 'male'] ["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived: ", rate_men, "\n")

from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId' : test_data.PassengerId, 'Survived' : predictions})
output.to_csv('submission.csv', index = False)
print("Your submission was successfully saved! \n")