from urllib.request import urlopen
import numpy as np
import requests
import pandas as pd
import json, time, datetime
import random
import math
import sklearn.preprocessing as prep
from tempfile import TemporaryFile

DATA_MARKET = 'data/poloniex/'
DATA_TWITTER = 'data/twitter/sentiment/'
DATA_BLOCKCHAIN = 'data/blockchain/'

INPUT_SEQ_LENGTH = 288 # 3*24*60/5
OUTPUT_SEQ_LENGTH = 24 # 4 hours

DROP_COLUMNS = ['ltc_close', 'ltc_volume', 'ltc_quoteVolume', 'eth_close', 'eth_volume', 'eth_quoteVolume', 'xrp_close', 'xrp_volume', 'xrp_quoteVolume']

USE_TWITTER = True
USE_BLOCKCHAIN = True

TARGET_VARIABLE = 'btc_close'

class PastSampler:

    def __init__(self, N, K, sliding_window = True, step_size=1):
        self.K = K
        self.N = N
        self.sliding_window = sliding_window
        self.step_size = step_size
 
    def transform(self, A):
        M = self.N + self.K     #Number of samples per row (sample + target)
        #indexes
        if self.sliding_window:
            I = np.arange(M) + np.arange(A.shape[0] - M + 1, step=self.step_size).reshape(-1, 1)
        else:
            if A.shape[0]%M == 0:
                I = np.arange(M)+np.arange(0,A.shape[0],M).reshape(-1,1)
                
            else:
                I = np.arange(M)+np.arange(0,A.shape[0] -M,M).reshape(-1,1)    
        #print(I)
        #print(I.shape)
        
        B = A[I].reshape(-1, M * A.shape[1], A.shape[2])
        ci = self.N * A.shape[1]    #Number of features per sample
        #print('ci', ci)
        #print('B shape', B.shape)
        return B[:, :ci], B[:, ci:, 0:1] #Sample matrix, Target matrix


def date_to_timestamp(s):
    return time.mktime(datetime.datetime.strptime(s, "%d/%m/%Y").timetuple())
    
def print_time(unix, msg=''):
    print(msg, time.ctime(int(unix)))

def split_data(data, s='01/03/2018'):
    split_time = date_to_timestamp(s)
    train = data.query('date<=@split_time')
    test = data.query('date>@split_time')
    return train, test

def download_data():
    # connect to poloniex's API
    CURRENCIES = ['USDT_BTC', 'USDT_LTC', 'USDT_ETH', 'USDT_XRP']
    url = 'https://poloniex.com/public?command=returnChartData&currencyPair=$C&start=1356998100&end=9999999999&period=300'
    urls = [url.replace('$C', c) for c in CURRENCIES]

    for i, c in enumerate(CURRENCIES):
        with urlopen(urls[i]) as url:
            r = url.read()
            d = json.loads(r.decode())
            df = pd.DataFrame(d)
            df = df.drop(columns=['high', 'low', 'open', 'weightedAverage'])
            #print(df.columns)
            df.to_pickle(DATA_MARKET + c + '.pkl')
            print('Successfully downloaded', c)
            print_time(min(df['date']), 'MIN:')
            print_time(max(df['date']), 'MAX:')
            
    
    df_btc = pd.read_pickle(DATA_MARKET + 'USDT_BTC.pkl')
    df_ltc = pd.read_pickle(DATA_MARKET + 'USDT_LTC.pkl')
    df_eth = pd.read_pickle(DATA_MARKET + 'USDT_ETH.pkl')
    df_xrp = pd.read_pickle(DATA_MARKET + 'USDT_XRP.pkl')
    
    
    #combine all dataframes into one with size of smallest dataframe - discard every other value
    count = [min(df_btc.count(numeric_only=True)), min(df_ltc.count(numeric_only=True)), min(df_eth.count(numeric_only=True)), min(df_xrp.count(numeric_only=True))]
    count = min(count)
    print_time(df_ltc['date'].iloc[-count], 'min date:')

    df_btc = df_btc.add_prefix('btc_')
    df_eth = df_eth.add_prefix('eth_')
    df_ltc = df_ltc.add_prefix('ltc_')
    df_xrp = df_xrp.add_prefix('xrp_')

    df_all = pd.concat([df_btc.iloc[-count:].reset_index(drop=True), df_eth.iloc[-count:].reset_index(drop=True), df_ltc.iloc[-count:].reset_index(drop=True), df_xrp.iloc[-count:].reset_index(drop=True)], axis=1)
    df_all.count(numeric_only=True)

    #cuz date column is same for every currency, we will discard others
    df_all.head()
    df_all['date'] = df_all['btc_date']
    df_all = df_all.drop(columns=['btc_date', 'ltc_date', 'eth_date', 'xrp_date'])
    df_all.to_pickle(DATA_MARKET + 'combined.pkl')

    
def load_data():
    """
    
    """    
    price_data = pd.read_pickle(DATA_MARKET + 'combined.pkl')

    currency = TARGET_VARIABLE.split('_')[0]

    sentiment_data = pd.read_pickle(DATA_TWITTER + currency + '_expanded.pkl')
    blockchain_data = pd.read_pickle(DATA_BLOCKCHAIN + currency + '_blockchain.pkl')

    #price_data = price_data.drop(columns=DROP_COLUMNS)
    
    min_date = min(sentiment_data['date'])
    max_date = max(sentiment_data['date'])
    
    price_data = price_data.query('@min_date <= date <= @max_date')
    data = price_data

    if USE_TWITTER:
        data = pd.merge(data, sentiment_data, how='inner', left_on='date', right_on='date')
    
    if USE_BLOCKCHAIN:
        data = pd.merge(data, blockchain_data, how='inner', left_on='date', right_on='date')
    
    return data

def normalize_fit_transform(X, fields=None):
    """
    Normalize data 
    """
    global scaler 
    scaler = prep.MinMaxScaler(feature_range=(0,1))
    if fields is not None:
        X = scaler.fit_transform(X[fields])
    else:
        X = scaler.fit_transform(X)
    return X, scaler

def normalize_transform(X):
    if scaler is None:
        print('Scaler doesnt exist, please use normalize_fit_transform function first')
    else:
        X = scaler.transform(X)
        return X
    
def denormalize_1d(data, min_, scale_):
    data -= min_
    data /= scale_
    return data

def denormalize_full(data):
    if scaler is None:
        print('Scaler doesnt exist, please use normalize_fit_transform function first')
    else:
        X = scaler.inverse_transform(data)
        return X

def fetch_batch_size_random(X, Y, batch_size):
    """
    Returns randomly an aligned batch_size of X and Y among all examples.
    The external dimension of X and Y must be the batch size (eg: 1 column = 1 example).
    X and Y can be N-dimensional.
    """
    assert X.shape[0] == Y.shape[0], (X.shape, Y.shape)
    idxes = np.random.randint(X.shape[0], size=batch_size)
    X_out = np.array(X[idxes]).transpose((1, 0, 2))
    Y_out = np.array(Y[idxes]).transpose((1, 0, 2))
    return X_out, Y_out

def fetch_batch_size_random_keras(X, Y, batch_size):
    assert X.shape[0] == Y.shape[0], (X.shape, Y.shape)
    idxes = np.random.randint(X.shape[0], size=batch_size)
    X_out = np.array(X[idxes])
    Y_out = np.array(Y[idxes])
    return X_out, Y_out

X_train = []
Y_train = []
X_test = []
Y_test = []

def prepare_data(input_seq_length, output_seq_length, sliding_window=True, step_size=5):
    data = load_data()
    cols = [TARGET_VARIABLE] + [col for col in data if col != TARGET_VARIABLE]
    data = data[cols]
    train, test = split_data(data)

    train = train.drop(columns=['date'])
    test = test.drop(columns=['date'])

    train, _ = normalize_fit_transform(train)
    test = normalize_transform(test)

    ps = PastSampler(input_seq_length, output_seq_length, sliding_window=True, step_size=step_size)

    X_train, Y_train = ps.transform(train[:,None,:])
    X_test, Y_test = ps.transform(test[:,None,:])
    
    return X_train, Y_train, X_test, Y_test

def generate_data_tf(isTrain, batch_size):
    """
    test
    """
    global Y_train
    global X_train
    global X_test
    global Y_test
    
    if len(Y_test) == 0:
        X_train, Y_train, X_test, Y_test = prepare_data(INPUT_SEQ_LENGTH, OUTPUT_SEQ_LENGTH, sliding_window=True, step_size=5)

    if isTrain:
        return fetch_batch_size_random(X_train, Y_train, batch_size)
    else:
        return fetch_batch_size_random(X_test,  Y_test,  batch_size)

def generate_data_keras_batch(isTrain, batch_size):
    global Y_train
    global X_train
    global X_test
    global Y_test
    
    if len(Y_test) == 0:
        X_train, Y_train, X_test, Y_test = prepare_data(INPUT_SEQ_LENGTH, OUTPUT_SEQ_LENGTH, sliding_window=True, step_size=5)

    if isTrain:
        return fetch_batch_size_random_keras(X_train, Y_train, batch_size)
    else:
        return fetch_batch_size_random_keras(X_test, Y_test, batch_size)

def generate_data_keras(input_seq_length, output_seq_length, step_size=5):
    INPUT_SEQ_LENGTH = input_seq_length
    OUTPUT_SEQ_LENGTH = output_seq_length
    X_train, Y_train, X_test, Y_test = prepare_data(input_seq_length, output_seq_length, sliding_window=True, step_size=step_size)
    return X_train, Y_train, X_test, Y_test