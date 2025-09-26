#importing libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

#importing dataset
df = pd.read_csv("heart.csv")

#independent variable
X = df.drop('target', axis=1)
#dependent variable
y = df['target']
#spliting the dataset into traning and testing of x and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#loading model into model named variable with max 1000 iteratons
model = LogisticRegression(max_iter=1000)
#giving data for traning
model.fit(X_train, y_train)
#dumping the pre-trained model in model.pkl file
joblib.dump(model, "model.pkl")





