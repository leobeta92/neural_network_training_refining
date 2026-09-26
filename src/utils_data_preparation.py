import numpy as np

import src.utils as utils

def validate_shape(x,y,data):
    print(f'{str.capitalize(data)}: X=%s, y=%s' % (x.shape, y.shape))

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

def create_validation_training_data(x_train, y_train):

    return {
        'x_val_scaled': x_train[-10000:],
        'y_val': y_train[-10000:],
        'x_train': x_train[:-10000],
        'y_train': y_train[:-10000]
    }
