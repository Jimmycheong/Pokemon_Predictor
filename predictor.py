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


pokemon = np.array([80,120,130])

pred = model.predict(pokemon)
print(pred)

pokemon2 = np.array([54,62,55])
pred = model.predict(pokemon2)
print(pred)


pokemon3 = np.array([32,36,27])
pred = model.predict(pokemon3)
print(pred)

# Test Jigglypuff 
jig_1 = np.array([[115, 45, 20]])
pred_jig = clf.predict(jig_1)
print(pred_jig)

# Test Pikachu 
pika = np.array([35, 55, 40])
pred_single = clf.predict(pika)
print(pred_single)

# Test Charmander
test_4 = np.array([[39, 52, 43]])
pred_single4 = clf.predict(test_4)
print(pred_single4)

# Test Pokemon 
# Test Jigglypuff 
jig_1 = np.array([[115, 45, 20]])
pred_jig = clf.predict(jig_1)
print(pred_jig)

# Test Pikachu 
pika = np.array([35, 55, 40])
pred_single = clf.predict(pika)
print(pred_single)

# Test Rapidash
test_9 = np.array([[63, 103, 70]])
pred_single4 = clf.predict(test_9)
print(pred_single4)


