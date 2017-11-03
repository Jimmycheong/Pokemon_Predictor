#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:15:50 2017

@author: kch31
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#pokemon_count 
pc = 100

'''
Pokemon Data Scaper and File writer
'''

dataset = pd.read_csv('pokemon/pokemon_stats_data.csv')

X_1 = dataset.iloc[:, 2].values.astype(str)
X_2 = dataset.iloc[:, 4:8].values

# Reshape X_1 
reshaped_X1 = np.reshape(X_1, [len(X_1), 1])

X = np.append(reshaped_X1, X_2, axis=1)
Y = dataset.iloc[:, 1].values.astype(str).tolist()


scraped_data = []


# Pokemons of Interest
# Trains on the first 150 pokemon
poi = Y[:151]

for i in range(0, len(X)):
    if Y[i] in poi:        
        combine = np.append(np.array(Y[i]), np.array(X[i]))
        scraped_data.append(combine)

converted = np.array(scraped_data)


'''
Data generator 
'''

list_data = converted.tolist()
print("Length of data list: ", len(list_data))

final_data_list = np.zeros([1,5])


for i in range(0, len(list_data)): 

    labels = np.full((pc,1), list_data[i][0])
    type_ = np.full((pc,1), list_data[i][1])

    # Randomise linear statistics and reshape 
    hp_stats = np.random.normal(int(list_data[i][3]), 0.9, pc).reshape((pc,1))
    attack_stats = np.random.normal(int(list_data[i][4]), 0.9, pc).reshape((pc,1))
    defence_stats = np.random.normal(int(list_data[i][5]), 0.9, pc).reshape((pc,1))

#    combined_features = np.append(np.append(hp_stats, attack_stats, axis=1), defence_stats, axis=1)
    combined_features = np.append(type_, hp_stats, axis=1)
    combined_features = np.append(combined_features, attack_stats, axis=1)
    combined_features = np.append(combined_features, defence_stats, axis=1)   
#    combined_features = np.append(type_, combined_features, axis=1)
    combined_data = np.append(labels, combined_features, axis=1)

    final_data_list = np.vstack((final_data_list, combined_data))

# Remove first array
final_data_list = np.delete(final_data_list, 0, 0)
print("\nFinal list: \n", final_data_list)

# Shuffle data 
np.random.shuffle(final_data_list)

print("Final shape: ", final_data_list.shape)

# Insert headers
headers = np.array(['Name','Type', 'HP', 'Attack', 'Defence'])
final_data_list = np.insert(final_data_list, 0, headers,axis=0)


'''
Save generated Pokemon data 
'''

np.savetxt("generated_dataset.csv", final_data_list, delimiter=',', fmt="%s")
