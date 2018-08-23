import glob, csv, sys, os, re, json, nltk
import time, datetime
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPool1D, Dropout, Dense, Activation
from keras.callbacks import CSVLogger

PATH = "data/sentiment_analysis/tweets/*.tsv"
PATH_SAVE = "data/sentiment_analysis/tweets_predicted"
test_date = "2018-06-01"
val_date = "2018-05-01"

stop_words = set(nltk.corpus.stopwords.words("english"))
ps = nltk.stem.PorterStemmer()


if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    from urllib import urlopen


def get_currency_prices():

    # connect to poloniex's API
    CURRENCIES = ['USDT_BTC', 'USDT_LTC', 'USDT_ETH', 'USDT_XRP']
    url = 'https://poloniex.com/public?command=returnChartData&currencyPair=$C&start=1451602800&end=1530568800&period=86400'
    urls = [url.replace('$C', c) for c in CURRENCIES]

    for i, c in enumerate(CURRENCIES):
        with urlopen(urls[i]) as url:
            r = url.read()
            d = json.loads(r.decode())
            df = pd.DataFrame(d)
            df.to_pickle('data/sentiment_analysis/poloniex/' + c + '.pkl')
            print('Successfully downloaded', c)


def get_labels(training_data):

    btc_price_path = 'data/sentiment_analysis/poloniex/USDT_BTC.pkl'
    if not os.path.exists(btc_price_path):
        get_currency_prices()
    btc_df = pd.read_pickle(btc_price_path)
    btc_df_values = btc_df.values

    labels = []
    for data in training_data:

        index = np.where((btc_df_values[:,1]>=data[5]-23*60*60) & (btc_df_values[:,1]<data[5]+60*60))[0]
        if btc_df_values[index+1,0] > btc_df_values[index, 0]:
            coef = [(btc_df_values[index,0]/btc_df_values[index+1,0])[0], 1]
        elif btc_df_values[index+1,0] < btc_df_values[index, 0]:
            coef = [(btc_df_values[index+1,0]/ btc_df_values[index,0])[0], -1]

        labels = np.append(labels, coef).reshape(-1, 2)

    return labels


def make_dictionary(data):
    dict = {}
    i = 1
    for row in data:
        for word in row[1].split():
            if word not in dict.keys():
                dict[word] = i
                i += 1
    return dict


def make_bag_of_words(data, dict):
    bag_of_words = []
    for row in data:
        tweet_rep = np.zeros(shape=(30), dtype=np.int64)
        i = 0
        for word in row[1].split():
            ind = dict.get(word)
            tweet_rep[i] = ind if ind else 0
            i += 1
            if i==30:
                break
        bag_of_words = np.append(bag_of_words, tweet_rep).reshape(-1,30)

    return bag_of_words


def build_model(input_dim, output_dim, input_len):

    model = Sequential()
    model.add(Embedding(input_dim, output_dim, input_length=input_len))
    model.add(Conv1D(filters=250, kernel_size=5, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    return model


def call_model(model, train_bag, train_labels, val_bag, val_labels):
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_bag, train_labels,
              batch_size=8,
              epochs=50,
              validation_data=(val_bag, val_labels),
              verbose=1,
              callbacks=[CSVLogger('nn_models/logger_sent.csv', append=True)])

    model.save('nn_models/sent.h5')


def cnn_process(train_data, val_data, test_data, train_labels, val_labels):

    dict = make_dictionary(train_data)

    train_bag = make_bag_of_words(train_data, dict)
    val_bag = make_bag_of_words(val_data, dict)
    test_bag = make_bag_of_words(test_data, dict)

    input_dim = (sorted(dict.values(), reverse=True))[0] + 1
    output_dim = 100
    input_len = len(train_bag[1, :])
    model = build_model(input_dim, output_dim, input_len)
    call_model(model, train_bag, train_labels, val_bag, val_labels)
    model.load_weights('nn_models/sent.h5')
    predicted = model.predict(test_bag)
    return predicted