
import numpy as np
import requests
import pandas as pd

import random
import math

DATA_MARKET = 'data/poloniex/'
DATA_TWITTER = 'data/twitter/sentiment/'

class PastSampler:

    def __init__(self, N, K, sliding_window = True):
        self.K = K
        self.N = N
        self.sliding_window = sliding_window
 
    def transform(self, A):
        M = self.N + self.K     #Number of samples per row (sample + target)
        #indexes
        if self.sliding_window:
            I = np.arange(M) + np.arange(A.shape[0] - M + 1).reshape(-1, 1)
        else:
            if A.shape[0]%M == 0:
                I = np.arange(M)+np.arange(0,A.shape[0],M).reshape(-1,1)
                
            else:
                I = np.arange(M)+np.arange(0,A.shape[0] -M,M).reshape(-1,1)    
        #print(I)
        print(I.shape)
        
        B = A[I].reshape(-1, M * A.shape[1], A.shape[2])
        ci = self.N * A.shape[1]    #Number of features per sample
        print('ci', ci)
        print('B shape', B.shape)
        return B[:, :ci], B[:, ci:] #Sample matrix, Target matrix

def load_data():
    """
    
    """    
    price_data = pd.read_pickle(DATA_MARKET + 'combined.pkl')
    sentiment_data = pd.read_pickle(DATA_TWITTER + 'btc_expanded.pkl')
    
    min_date = min(sentiment_data['date'])
    max_date = max(sentiment_data['date'])
    
    price_data = price_data.query('@min_date <= date <= @max_date')
    
    return pd.merge(price_data, sentiment_data, how='inner', left_on='date', right_on='date')

def normalize(X, Y=None):
    """
    Normalise X and Y according to the mean and standard deviation of the X values only.
    """
    # # It would be possible to normalize with last rather than mean, such as:
    # lasts = np.expand_dims(X[:, -1, :], axis=1)
    # assert (lasts[:, :] == X[:, -1, :]).all(), "{}, {}, {}. {}".format(lasts[:, :].shape, X[:, -1, :].shape, lasts[:, :], X[:, -1, :])
    mean = np.expand_dims(np.average(X, axis=1) + 0.00001, axis=1)
    stddev = np.expand_dims(np.std(X, axis=1) + 0.00001, axis=1)
    print (mean.shape, stddev.shape)
    # print (X.shape, Y.shape)
    X = X - mean
    X = X / (2.5 * stddev)
    if Y is not None:
        assert Y.shape == X.shape, (Y.shape, X.shape)
        Y = Y - mean
        Y = Y / (2.5 * stddev)
        return X, Y
    return X

def fetch_batch_size_random(X, Y, batch_size):
    """
    Returns randomly an aligned batch_size of X and Y among all examples.
    The external dimension of X and Y must be the batch size (eg: 1 column = 1 example).
    X and Y can be N-dimensional.
    """
    assert X.shape == Y.shape, (X.shape, Y.shape)
    idxes = np.random.randint(X.shape[0], size=batch_size)
    X_out = np.array(X[idxes]).transpose((1, 0, 2))
    Y_out = np.array(Y[idxes]).transpose((1, 0, 2))
    return X_out, Y_out

def generate_data(isTrain, batch_size):
    """
    test
    """
    # 40 pas values for encoder, 40 after for decoder's predictions.
    input_seq_length = 864 # 3 Days - 3*24*60/5
    output_seq_length = 24 # 2 hours
    split = 0.85

    global Y_train
    global X_train
    global X_test
    global Y_test
    # First load, with memoization:
    if len(Y_test) == 0:
        # Dejan
        data = load_data()
        ps = PastSampler(input_seq_length, output_seq_length, sliding_window=True)

        # All data, aligned:
        X, Y = ps.transform(data.as_matrix()[:,None,:])
        #X, Y = normalize(X, Y)

        # Split 85-15:
        X_train = X[:int(len(X) * split)]
        Y_train = Y[:int(len(Y) * split)]
        X_test = X[int(len(X) * split):]
        Y_test = Y[int(len(Y) * split):]

    if isTrain:
        return fetch_batch_size_random(X_train, Y_train, batch_size)
    else:
        return fetch_batch_size_random(X_test,  Y_test,  batch_size)

def gen_data(input_seq_length, output_seq_length):
    data = load_data()
    ps = PastSampler(input_seq_length, input_seq_length, sliding_window=False)
    X, Y = ps.transform(data.as_matrix()[:,None,:])
    return X, Y




