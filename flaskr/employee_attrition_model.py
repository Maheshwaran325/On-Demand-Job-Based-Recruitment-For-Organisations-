import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score

# Load the employee data into a Pandas dataframe
data = pd.read_csv("train_data.csv")

# Encode the categorical data
encoder = OneHotEncoder(handle_unknown='ignore')
X = data.drop(["Attrition", "EmployeeID"], axis=1)
X_encoded = pd.DataFrame(encoder.fit_transform(X[X.columns[X.dtypes == 'object']]).toarray())

# Convert feature names to strings
X_encoded.columns = X_encoded.columns.astype(str)

X = X.drop(X.columns[X.dtypes == 'object'], axis=1)
X = pd.concat([X, X_encoded], axis=1)

y = data["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a Random Forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model,"model.pkl")
joblib.dump(encoder, "encoder.pkl")

# scores = cross_val_score(model, X, y, cv=10)
                         
# y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, pos_label='Yes')
# recall = recall_score(y_test, y_pred, pos_label='Yes')
# f1 = f1_score(y_test, y_pred, pos_label='Yes')
# conf_matrix = confusion_matrix(y_test, y_pred, labels=['No', 'Yes'])

# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1-Score:", f1)
# print("Confusion Matrix:\n", conf_matrix)