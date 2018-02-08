'''
Hello there

'''
import pandas as pd
import numpy as np
import tensorflow as tf

def get_pokemon_data_from_csv(path):
    data = []
    
    dataset = pd.read_csv(path)
    type_column = dataset.iloc[:, 2].values.astype(str)
    reshaped_type_column = np.reshape(type_column, [len(type_column), 1]) # Reshape to [?, 1]
    stats_columns = dataset.iloc[:, 4:8].values

    features = np.append(reshaped_type_column, stats_columns, axis=1)
    raw_labels = dataset.iloc[:, 1].values.astype(str)
    labels = [x.lower() for x in raw_labels]

    for i in range(0, len(features)):
        combine = np.append(np.array(labels[i]), np.array(features[i]))
        data.append(combine.tolist())

    return np.array(data)

def get_pokemon_of_interest_data(pokemon_array, poi):
    
    '''
    Params: 
        pokemon_array(list): A list of data containing all pokemon stats
        poi(list) : A list of names of pokemon of interest

    Returns:
        poi_data (list) = A list of lists containing pokemon of interest data

    '''

    poi_data = []
    for i in range(0, len(pokemon_array)-1):
        if pokemon_array[i][0] in poi: 
            poi_data.append(pokemon_array[i])
    
    return poi_data

def breed_pokemon(poi_data, breed_count):

    '''
    poi_data(list): A list of lists containing pokemon data
    breed_count(int): A quantity of pokemon to breed per poi_data sample

    Returns:
        A list of 

    '''

    # Create an zero-filled array with correct shape
    return_data = np.zeros([1,4])

    for i in range(0, len(poi_data)): 

        labels = np.full((breed_count,1), poi_data[i][0])

        # Randomise linear statistics and reshape 
        hp_stats = np.random.normal(int(poi_data[i][3]), 0.9, breed_count).reshape((breed_count,1))
        attack_stats = np.random.normal(int(poi_data[i][4]), 0.9, breed_count).reshape((breed_count,1))
        defence_stats = np.random.normal(int(poi_data[i][5]), 0.9, breed_count).reshape((breed_count,1))

        # Stitch stats together
        combined_features = np.append(hp_stats, attack_stats, axis=1)
        combined_features = np.append(combined_features, defence_stats, axis=1)   
        combined_data = np.append(labels, combined_features, axis=1)
        return_data = np.vstack((return_data, combined_data))

    return_data = np.delete(return_data, 0, 0) # Remove first array
    print("\nFinal list: \n", return_data)

    return return_data

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
