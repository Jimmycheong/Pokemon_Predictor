#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:15:50 2017

@author: kch31
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from functions import get_pokemon_of_interest_data, get_pokemon_data_from_csv, breed_pokemon

DATA_FILEPATH = 'resources/pokemon_stats_data.csv'
SAVE_FILEPATH = 'resources/generated_dataset.csv'

#pokemon_count 
BREED_COUNT = 3

def main():

    # Read from file and generate a list of pokemon data
    pokemon_data = get_pokemon_data_from_csv(DATA_FILEPATH)

    # Create a list of pokemon of interest
    pokemon_of_interest = pd.DataFrame(pokemon_data).iloc[:,0][:5].tolist()

    pokemon_of_interest = ['Gyarados', "Onix", "Mew", 'Snorlax']

    pokemon_of_interest = [x.lower() for x in pokemon_of_interest]

    # Grab data for the pokemon of interest
    poi_data = get_pokemon_of_interest_data(pokemon_data, pokemon_of_interest)

    print("Number of pokemon of interest: ", len(poi_data))

    bred_pokemon_data = breed_pokemon(poi_data, BREED_COUNT)

    # Shuffle data 
    np.random.shuffle(bred_pokemon_data)

    print("Final shape: ", bred_pokemon_data.shape)

    # Insert headers
    headers = np.array(['Name', 'HP', 'Attack', 'Defence'])
    bred_pokemon_data = np.insert(bred_pokemon_data, 0, headers,axis=0)

    # Save generated Pokemon data
    np.savetxt(SAVE_FILEPATH, bred_pokemon_data, delimiter=',', fmt="%s")

if __name__ == '__main__':
    main()
