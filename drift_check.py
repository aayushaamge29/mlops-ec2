import numpy as np

TRAIN_MEAN = [5.8, 3.0, 3.7, 1.1]

def check_drift(input_features):
    current_mean = np.mean(input_features)
    train_mean = np.mean(TRAIN_MEAN)

    if abs(current_mean - train_mean) > 1.0:
        return True
    return False
