"""
batchData.py
descripition: Loads data from specified numpy array, shuffles and splits data into
training and test sets. Generates batches of data based on the data length, batch size
and number of epochs.
"""
import numpy as np

# Generating batches of data
def batch_iter(data, batch_size, num_epochs, shuffle=True):

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_data(x_file, y_file, useEvaluationSet=True):
    # load negative and positive training set
    # words replaced by word ids referencing vocabulary
    x_d = np.load(x_file)
    y_d = np.load(y_file);
    # set random seed number (for reproducibility)
    np.random.seed(10)
    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y_d)))
    x_shuffled = x_d[shuffle_indices]
    y_shuffled = y_d[shuffle_indices]

    test_ratio = int(len(y_d)*0.05);
    # Split train/test set
    if useEvaluationSet == True:
        x_train, x_dev = x_shuffled[:-test_ratio], x_shuffled[-test_ratio:]
        y_train, y_dev = y_shuffled[:-test_ratio], y_shuffled[-test_ratio:]
        return [x_train, y_train, x_dev, y_dev];
    else:
        x_train = x_shuffled;
        y_train = y_shuffled;
        return [x_train, y_train];
