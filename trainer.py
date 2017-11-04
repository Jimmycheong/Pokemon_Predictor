'''
Preprocessing 
'''

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import pickle 

'''
Load data from file
'''

# Load Data from csv file
dataset = pd.read_csv('generated_dataset.csv')

X = dataset.iloc[:, 1:].values.astype(str)
Y = dataset.iloc[:, 0].values.astype(str)


'''
Encode data
'''

# Label Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0]).astype(int)


# One hot encoding
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

# Save encoders
with open("encoders/labelencoder.pkl", "wb") as file:
    pickle.dump(labelencoder, file)

with open("encoders/onehotencoder.pkl", "wb") as file:
    pickle.dump(onehotencoder, file)

'''
Partition data into training and testing sets
'''

# Split training and testing data
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size = 0.2, random_state=43)


# Feature Scaling
# from sklearn.preprocessing import MinMaxScaler 
# min_max_scaler = MinMaxScaler()
# X_train = min_max_scaler.fit_transform(X_train)
# X_test = min_max_scaler.fit_transform(X_test)


'''
Training 
 
'''

from sklearn.svm import SVC
clf = SVC(kernel='rbf')

clf.fit(X_train, y_train)

with open("pokemon_model.pkl", 'wb') as file: 
    pickle.dump(clf, file)
    

'''
Testing 
'''

pred = clf.predict(X_test).astype(str)

from sklearn.metrics import accuracy_score 
acc = accuracy_score(pred, y_test)

print("Accuracy of training data", acc)



