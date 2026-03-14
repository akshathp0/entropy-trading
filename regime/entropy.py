import numpy as np
import pandas as pd
from regime import matrix
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

EXPANDING_WINDOW_MINIMUM = config['expanding_window_minimum']
STATE_COUNT_MINIMUM = config['state_count_minimum']

def find_entropy_vector(transition_matrix):
    transition_matrix = transition_matrix.replace(0, np.nan)
    entropy = -(transition_matrix * np.log(transition_matrix)).sum(axis = 1)

    return entropy

def build_stationary_entropy(df, vector):
    df['Stationary Entropy'] = df['State'].map(vector)

    return df

def build_expanding_entropy(df, start = EXPANDING_WINDOW_MINIMUM, count_min = STATE_COUNT_MINIMUM):
    expanding = [np.nan] * len(df)

    for i in range(start, len(df)):
        window_df = df.iloc[:i]
        transition_matrix, counts = matrix.matrix_template(window_df)

        vector = find_entropy_vector(transition_matrix)
        state = df['State'].iat[i]
        fallback_entropy = vector.mean()

        if state not in vector.index:
            expanding[i] = fallback_entropy
        else:
            state_count = counts.loc[state].sum()

            if state_count < count_min:
                expanding[i] = fallback_entropy
            else:
                expanding[i] = vector.loc[state].item()

    df['Expanding Entropy'] = expanding

    return df