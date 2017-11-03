import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import pickle 

'''
Preprocessing 
'''


dataset = pd.read_csv('generated_dataset.csv')

X = dataset.iloc[:, 1:].values
Y = dataset.iloc[:, 0].values.astype(str)



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



