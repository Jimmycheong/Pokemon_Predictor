#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:26:44 2017

@author: kch31
"""

'''
Testing 
'''

import pickle 
import numpy as np


with open("pokemon_model.pkl", "rb") as file: 
    model = pickle.load(file)

with open("encoders/labelencoder.pkl", "rb") as file: 
    labelencoder = pickle.load(file)

with open("encoders/onehotencoder.pkl", "rb") as file: 
    onehotencoder = pickle.load(file)

'''
Predict Pokemon based on inputs
'''

pokemon_stats = np.array([['Grass', 80, 82, 83]])

# Label Encode
pokemon_stats[:, 0] = labelencoder.transform(pokemon_stats[:, 0])

# One Hot Encode
pokemon_stats = onehotencoder.transform(pokemon_stats).toarray()

# Predict Pokemon
prediction = model.predict(pokemon_stats)

print(prediction)