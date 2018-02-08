import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing

from functions import(
    get_pokemon_data_from_csv,
    train_input_fn,
    eval_input_fn
)

DATA_PATH = 'resources/generated_dataset.csv'
TRAIN_STEPS = 1000
BATCH_SIZE = 100
CSV_COLUMN_NAMES = ["Name", "HP","Attack", "Defence"]
DATA_SPLIT_RATIO = 0.8


'''
Data Preparation
'''

df = pd.read_csv(DATA_PATH, names=CSV_COLUMN_NAMES, header=0)

# Replace non-numeric name values with encoded values
le = preprocessing.LabelEncoder()
encoded_labels = le.fit_transform(df.iloc[:, 0].tolist())
POKEMON_NAMES = le.classes_.tolist()
df.iloc[:,0] = encoded_labels

# Split data
train_df, test_df = df[:int(len(df)*DATA_SPLIT_RATIO)], df[int(len(df)*DATA_SPLIT_RATIO):]

# Extract features and labels
train_x, train_y = train_df, train_df.pop("Name")
test_x, test_y = test_df, test_df.pop("Name")

feature_columns = []
for key in train_x.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))
    

'''
Training & Evaluation
'''

classifier = tf.estimator.DNNClassifier(
                feature_columns=feature_columns,
                hidden_units=[10, 10],         # Two hidden layers of 10 nodes each.
                n_classes=len(le.classes_),
                model_dir="models/pokemon_predictor"
        )    # Number of distinct labels.

classifier.train(
        input_fn=lambda:train_input_fn(train_x, train_y, BATCH_SIZE),
        steps=8)

eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(
        test_x, 
        test_y,
        BATCH_SIZE
    )
)

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

'''
Prediction
'''

expected = ['mew', 'gyrados', 'onix']
predict_x = {
    'HP': [100, 95, 34],
    'Attack': [100, 125, 44],
    'Defence': [100, 80, 160],
}

predictions = classifier.predict(
        input_fn=lambda:eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=BATCH_SIZE))


for pred_dict, expec in zip(predictions, expected):
    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print(template.format(POKEMON_NAMES[class_id],
                          100 * probability, expec))
