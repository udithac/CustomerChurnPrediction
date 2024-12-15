import pandas as pd 
from sklearn.model_selection import train_test_split
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

df = pd.read_csv('customerdata.csv')

print(df.head())

X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis =1))
y = df['Churn'].apply(lambda x: 1 if x=='Yes' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

X_train.head()
y_train.head()

#import dependancies

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

#Build the Model
model =  Sequential()
model.add(Dense(units = 32, activation= 'relu', input_dim = len(X_train.columns)))
model.add(Dense(units = 64, activation= 'relu'))
model.add(Dense(units=1, activation='sigmoid'))


#Compile the Model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()
#Fit the Model
model.fit(X_train, y_train, epochs=50, batch_size=32)

#Predictions
y_hat = model.predict(X_test)
y_hat = [0 if val<.5 else 1 for val in y_hat]

print("y_hat: " ,y_hat)

print("Accuracy Score: ",accuracy_score(y_test, y_hat))

#Save the Model
model.save('cs_churn_tfmodel.keras')
