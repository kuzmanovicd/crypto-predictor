import glob, csv, sys, os, re
import time, datetime
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPool1D, Dropout, Dense, Activation
from keras.callbacks import CSVLogger
import sklearn.preprocessing as prep

PATH = "data/tweets/influencers/*.tsv"
PATH_SAVE = "data/tweets/predicted"



if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    from urllib import urlopen

def take_tweets(data, all_data):
    for d in data:
        text = d[1].lower()
        # if ("bitcoin" or "btc" or "litecoin" or "ltc" or "ethereum" or "etc" or "ripple" or "xrp") in text:
        if ("bitcoin" or "btc") in text:
            all_data = np.append(all_data, d).reshape(-1,11)
    return all_data


def take_files(FILE_PATH):
    all_data = []                           #taking tweets from influencers, and append if bitcoin is mentioned
    for path in sorted(glob.glob(FILE_PATH)):
        with open(path, 'r') as f:
            data = [row for row in csv.reader(f.read().splitlines(), delimiter='\t')]
            all_data = take_tweets(data, all_data)
        print(path)
    return all_data


def get_data():                     #divide data to training, validating and test data

    #all_data = take_files(PATH)
    path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'temp/files.npy')
    #np.save(path, all_data)
    all_data = np.load(path)
    all_data_size = len(all_data)
    training_data = all_data[:int(all_data_size*0.9),:]
    validating_data = all_data[int(all_data_size*0.9):,:]
    return training_data, validating_data


def get_currency_prices():

    # connect to poloniex's API
    CURRENCIES = ['USDT_BTC', 'USDT_LTC', 'USDT_ETH', 'USDT_XRP']
    url = 'https://poloniex.com/public?command=returnChartData&currencyPair=$C&start=1451692800&end=1519862400&period=86400'
    urls = [url.replace('$C', c) for c in CURRENCIES]
    #
    # for i, c in enumerate(CURRENCIES):
    #     with urlopen(urls[i]) as url:
    #         r = url.read()
    #         d = json.loads(r.decode())
    #         df = pd.DataFrame(d)
    #         #print(df.columns)
    #         df.to_pickle('data/tweets/poloniex/' + c + '.pkl')
    #         print('Successfully downloaded', c)


def get_labels(training_data, validating_data):               #get normalized prices for specific tweet on specific date

    btc_df = pd.read_pickle('data/tweets/poloniex/USDT_BTC.pkl')
    btc_df_values = btc_df.values

    training_labels = []
    index = 0
    br1 = 0
    br2 = 0
    for data in training_data:
        index = np.where((btc_df_values[:,1]<int(data[5])+86400) & (btc_df_values[:,1]>int(data[5])))[0]
        if btc_df_values[index+1,0] > btc_df_values[index, 0]:
            coef = [(btc_df_values[index,0]/btc_df_values[index+1,0])[0], 1]
            br1 += 1
        else:
            coef = [(btc_df_values[index+1,0]/ btc_df_values[index,0])[0], -1]
            br2 += 1
        training_labels = np.append(training_labels, coef).reshape(-1, 2)

    validating_labels = []
    for data in validating_data:
        index = np.where((btc_df_values[:, 1] < int(data[5]) + 86400) & (btc_df_values[:, 1] > int(data[5])))[0]
        if btc_df_values[index + 1, 0] > btc_df_values[index, 0]:
            coef = [(btc_df_values[index, 0] / btc_df_values[index + 1, 0])[0], 1]
            br1 += 1
        else:
            coef = [(btc_df_values[index + 1, 0] / btc_df_values[index, 0])[0], -1]
            br2 +=1
        validating_labels = np.append(validating_labels, coef).reshape(-1, 2)

    start_price = int(btc_df_values[index, 0])
    start_time = int(btc_df_values[index, 1])

    print(br1)
    print(br2)

    return training_labels, validating_labels, start_price, start_time


def prepare_text(data):                 #prepare text for text processing

    i = 0
    for text in data[:,1]:
        text = text.lower()
        text = re.sub(r'@\w+', '', text)             #remove mentions
        text = re.sub(r'@(\s+)\w+', '', text)
        text = re.sub(r'http\S+', '', text)         #remove links
        text = re.sub(r'\w*\\.\w*', '', text)
        text = re.sub(r'/\w*', '', text)
        text = re.sub(r'([^\s\w]|_)+', '', text)    #only alfanumeric and space
        text = re.sub(r'\W*\b\w{18,60}\b', '', text)        #remove big words
        if text!='':
            data[i,1] = text
            i += 1

    return data


def make_dictionary(data):
    dict = {}
    i = 1
    for text in data[:,1]:
        for word in text.split():
            if word not in dict.values():
                dict[word] = i
                i += 1
    return dict


def make_bag_of_words(data, dict):
    bag_of_words = []

    # x = 0
    for tweet in data[:,1]:
        # if len(re.findall("[a-zA-Z_]+", tweet))>x:
        #     x = len(re.findall("[a-zA-Z_]+", tweet))

        tweet_rep = np.zeros(shape=(30), dtype=np.int64)
        i = 0
        for word in tweet.split():
            ind = dict.get(word)
            tweet_rep[i] = ind if ind else 0
            i += 1
            if i==30:
                break

        bag_of_words = np.append(bag_of_words, tweet_rep).reshape(-1,30)

    # print(x)
    return bag_of_words


def build_model(input_dim, output_dim, input_len):

    model = Sequential()
    model.add(Embedding(input_dim, output_dim, input_length=input_len))
    model.add(Conv1D(filters=250, kernel_size=3, padding='valid', activation='relu', strides=1))
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
              epochs=5,
              validation_data=(val_bag, val_labels),
              verbose=1,
              callbacks=[CSVLogger('nn_models/logger_sent.csv', append=True)])

    model.save('nn_models/sent.h5')


def check_results(curr_time, curr_price, predicted):
    num_tweets = 0
    coef_tweets = 0
    i = 0
    finall_coef = {}

    for data in test_data_p:
        print(predicted[i])
        if data[5].isdigit():
            data_time = int(data[5])
        else:
            data_time = int(time.mktime(datetime.datetime.strptime(data[5], "%Y-%m-%d %H:%M:%S").timetuple()))

        if (data_time < curr_time):
            num_tweets += 1
            if predicted[1][i] > 0:
                coef_tweets += predicted[0][i]
            else:
                coef_tweets -= predicted[0][i]
        else:
            if num_tweets != 0:
                coef_tweets /= num_tweets

            finall_coef[curr_time] = coef_tweets
            num_tweets = 0
            coef_tweets = 0
            curr_time += 86400

        i += 1

    finall_prices = {}
    for key, value in finall_coef.items():
        if value > 0:
            curr_price = curr_price / value
        else:
            curr_price = curr_price * abs(value)

        finall_prices[key] = curr_price

    return finall_coef, finall_prices


def save_results(results, FILE_PATH):

    i = 0
    for path in sorted(glob.glob(FILE_PATH)):
        with open(path, 'r') as f:
            all_data = []
            data = [row for row in csv.reader(f.read().splitlines(), delimiter='\t')]
            for d in data:
                text = d[1].lower()
                if ("bitcoin" or "btc") in text:
                    d = d[:-1]
                    d.extend(results[i])
                    all_data = np.append(all_data, d).reshape(-1, 15)
                    i += 1

        print(path)

        with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['currency', 'tweet', 'username', 'favorites', 'retweets', 'date', 'id', 'permalink',
                             'geo', 'hashtags', 'mentions', 'lex_sent', 'vader_sent', 'cnn_sent', 'cnn_pos_neg'])

            for d in all_data:
                writer.writerow(d)


def save_test_results(predicted, FILE_PATH):
    model.load_weights('nn_models/sent.h5')
    predicted = model.predict(test_bag)

    save_results(predicted, FILE_PATH)


if __name__ == '__main__':


    train_data, val_data = get_data()
    test_data = take_files("data/tweets/test_tweets/*.tsv")
    train_data_p = prepare_text(train_data)
    val_data_p = prepare_text(val_data)
    test_data_p = prepare_text(test_data)

    train_labels, val_labels, curr_price, curr_time = get_labels(train_data_p, val_data_p)

    #save_results(train_labels, "data/tweets/predicted/*.tsv")

    dict = make_dictionary(train_data_p)

    train_bag = make_bag_of_words(train_data_p, dict)
    val_bag = make_bag_of_words(val_data_p, dict)
    test_bag = make_bag_of_words(test_data_p, dict)

    input_dim = (sorted(dict.values(), reverse=True))[0]+1
    output_dim = 100
    input_len = len(train_bag[1,:])
    model = build_model(input_dim, output_dim, input_len)
    call_model(model, train_bag, train_labels, val_bag, val_labels)
    model.load_weights('nn_models/sent.h5')
    predicted = model.predict(test_bag)
    check_results(curr_time, curr_price, predicted)
    #save_test_results(predicted, "data/tweets/test_predicted/*.tsv")