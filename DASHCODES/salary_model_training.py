import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

sal = pd.read_csv('./sal.csv',header=0, index_col=None)
X = sal[['x']]
y = sal['y']
#spliting
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

#train the model
lm = LinearRegression() 
lm.fit(x_train,y_train)
filename = './salary.pkl'
joblib.dump(lm, filename)
joblib.dump(x_train, "training_data.pkl")
joblib.dump(y_train, "training_labels.pkl")

#test the model
filename = './salary.pkl'
loaded_model=joblib.load(filename)
y=loaded_model.predict([[21]])[0]
print(y)