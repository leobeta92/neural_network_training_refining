import numpy as np

import src.utils as utils

def create_train_test(x_train_rows, x_test_rows):
    x_train =  []
    x_test =  []

    for i in range(x_train_rows.shape[0]):
        x_train.append(utils.parse_record(x_train_rows[i]))

    for i in range(x_test_rows.shape[0]):
        x_test.append(utils.parse_record(x_test_rows[i]))

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    return ({
        'x_train': x_train, 
        'x_test': x_test
        })

def scale_values(x_train, x_test):
    return ({
        'x_train_scaled': x_train / 255.0,
        'x_test_scaled': x_test / 255.0
    })