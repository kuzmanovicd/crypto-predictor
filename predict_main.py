from urllib.request import urlopen
import json, datetime, time, csv, glob
import pandas as pd
import numpy as np
import sklearn.preprocessing as sk
from keras.models import Sequential
from keras.layers import CuDNNGRU, Dense, Dropout
from keras.callbacks import History, ModelCheckpoint, RemoteMonitor, CSVLogger, TensorBoard
import matplotlib.pyplot as plt
import pickle
import h5py

url_blokchain = 'https://api.blockchain.info/charts/$C?timespan=3years&sampled=false&format=csv'
url_price = 'https://poloniex.com/public?command=returnChartData&currencyPair=$C&start=1356998100&end=9999999999&period=300'

blokchain_features = [
            #'transactions-per-second',
            'avg-block-size',
            'cost-per-transaction',
            'difficulty',
            'hash-rate',
            'market-cap',
            'median-confirmation-time',
            'transaction-fees',
            'transaction-fees-usd',
            'n-transactions-per-block',
            'miners-revenue',
            'n-unique-addresses',
            'n-transactions',
            'n-transactions-total',
            #'mempool-growth',
            #'mempool-count',
            #'mempool-size',
            'n-transactions-excluding-popular',
            'n-transactions-excluding-chains-longer-than-100',
            'output-volume',
            'estimated-transaction-volume',
            'estimated-transaction-volume-usd']

currencies = ['USDT_BTC', 'USDT_ETH', 'USDT_XRP', 'USDT_LTC']
save_path = 'data/sentiment_analysis/price_info/'
tweets_path = 'data/sentiment_analysis/tweets_predicted/*.tsv'
start_date = "2016-01-01 00:00:00"
val_date = "2018-05-01 00:00:00"
test_date = "2018-06-01 00:00:00"
end_date = "2018-07-01 00:00:00"
period = 60*60
input_timestep = 24
output_timestep = 2
units = 96

def to_timestamp(str_time):
    return int(time.mktime(datetime.datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S').timetuple()))


def save_data():
    price_df_all = pd.DataFrame()
    for currency in currencies:
        url = url_price.replace('$C', currency)
        with urlopen(url) as u:
            price_str = u.read().decode()
            price_list = json.loads(price_str)
            price_df = pd.DataFrame.from_dict(price_list)
            price_df.to_pickle(save_path + str(currency) + '.pkl')
            currency_str = currency.split('_')[1].lower() + '_'
            price_df = price_df.add_prefix(currency_str)
            price_df['date'] = price_df[currency_str+'date']
            del price_df[currency_str+'date']
            if price_df_all.empty:
                price_df_all = price_df
            else:
                price_df_all = pd.merge(price_df_all, price_df, how='inner', left_on='date', right_on='date')
    price_df_all.to_pickle(save_path+'price_combined.pkl')

    blokchain_df = pd.DataFrame()
    for feature in blokchain_features:
        url = url_blokchain.replace('$C', feature)
        with urlopen(url) as u:
            blokchain_str = u.read().decode()
            blokchain_list = blokchain_str.split('\n')
            blokchain = dict()
            blokchain['date'] = []
            blokchain[feature] = []
            try:
                for node in blokchain_list:
                    nodes = node.split(',')
                    blokchain['date'].append(to_timestamp(nodes[0]))
                    blokchain[feature].append(nodes[1])
            except:
                print ('end of file')

            blokchain_df_current = pd.DataFrame(data=blokchain)
            if blokchain_df.empty:
                blokchain_df = blokchain_df_current
            else:
                blokchain_df = pd.merge(blokchain_df, blokchain_df_current, how='inner', left_on='date', right_on='date')

    blokchain_df.to_pickle(save_path + 'blockchain.pkl')

    tweets_list = []
    for path in glob.glob(tweets_path):
        with open(path, 'r') as csvfile:
            tweets = [tweet for tweet in csv.reader(csvfile, delimiter='\t') if tweet[0]!='currency']
            tweets_list.extend(tweets)

    tweets_df = pd.DataFrame.from_dict(tweets_list)
    tweets_df = tweets_df.rename(columns={0:'currency', 1:'tweet', 2:'ussername', 3:'favourites',
                                                       4:'retweets', 5:'date', 6:'id', 7:'permalink', 8:'geo',
                                                       9: 'hashtags', 10: 'mentions', 11: 'lex_sent', 12: 'vad_sent',
                                                       13: 'vad_pol', 14: 'cnn_sent', 15: 'cnn_pol'})
    tweets_df.to_pickle(save_path + 'tweets.pkl')


def fill_first_frame(dataframe):
    date = dataframe.at[0, 'date']
    dataframe.loc[0] = pd.Series(0, index=dataframe.columns.values)
    dataframe.at[0, 'date'] = date
    dataframe.at[0, 'round_interval'] = True
    return dataframe


def expand_data(dataframe, name):
    start_date_timestamp = to_timestamp(start_date)+60*60
    end_date_timestamp = to_timestamp(end_date)+60*60
    inverval = int((end_date_timestamp-start_date_timestamp)/period)
    dataframe = dataframe.query('date>=@start_date_timestamp & date<=@end_date_timestamp')
    dataframe['round_interval'] = False
    for row in dataframe.iterrows():
        index = row[0]
        if dataframe.at[index, 'date'] % period == 0:
            dataframe.at[index, 'round_interval'] = True

    interpolated = pd.DataFrame(index=range(0, inverval), columns=dataframe.columns.values)
    interpolated['date'] = range(start_date_timestamp, end_date_timestamp, period)
    interpolated['round_interval'] = True
    dataframe_expanded = pd.concat([interpolated, dataframe]).sort_values('date').reset_index(drop=True)
    dataframe_expanded = fill_first_frame(dataframe_expanded)
    dataframe_expanded = dataframe_expanded.astype('float64')
    dataframe_expanded['round_interval'] = dataframe_expanded['round_interval'].astype('bool')
    dataframe_expanded = dataframe_expanded.interpolate(method='linear', axis=0)
    dataframe_expanded = dataframe_expanded.query('round_interval')
    dataframe_expanded = dataframe_expanded.drop_duplicates(subset='date', keep='last').reset_index(drop=True)
    dataframe_expanded['date'] = dataframe_expanded['date'].astype('int64')
    del dataframe_expanded['round_interval']
    dataframe_expanded.to_pickle(save_path + name + '_expanded.pkl')
    return dataframe_expanded


def prepare_data():
    # price_data = pd.read_pickle(save_path + 'USDT_BTC.pkl')
    # price_data_expanded = expand_data(price_data, 'price')
    price_data_expanded = pd.read_pickle(save_path + 'price_expanded.pkl')

    blokchain_data = pd.read_pickle(save_path + 'blockchain.pkl')
    blokchain_data_expanded = expand_data(blokchain_data, 'blockchain')
    merged_price_blokchain = pd.merge(price_data_expanded, blokchain_data_expanded, how='inner', left_on='date', right_on='date')

    tweets_data = pd.read_pickle(save_path + 'tweets.pkl')
    columns = ['favourites', 'retweets', 'date', 'lex_sent', 'vad_sent', 'vad_pol', 'cnn_sent', 'cnn_pol']
    tweets_data = tweets_data[columns]
    tweets_data = tweets_data.astype('float64')
    tweets_data_expanded = expand_data(tweets_data, 'tweets')
    merged_data = pd.merge(merged_price_blokchain, tweets_data_expanded, how='inner', left_on='date', right_on='date')
    merged_data.to_pickle(save_path + 'merged_data.pkl')
    return merged_data


def get_index_of_test_data():
    start_date_timestamp = to_timestamp(start_date)
    test_date_timestamp = to_timestamp(test_date)
    test_index = int((test_date_timestamp - start_date_timestamp) / (60 * 60)) - 1
    return test_index


def generate_keras_data(merged_data, input_timestep, output_timestep):

    dataset_size = merged_data.shape[0]
    timestep_size = input_timestep+output_timestep
    keras_dataset_size = dataset_size - timestep_size + 1
    index = np.arange(0, timestep_size) + np.arange(keras_dataset_size).reshape(-1,1)
    keras_data = np.array(merged_data)[:,None,:]
    keras_data = keras_data[index].reshape(-1, timestep_size, merged_data.shape[1])

    test_index = get_index_of_test_data()
    train_data = keras_data[:test_index, :input_timestep, :]
    train_label = keras_data[:test_index, input_timestep:, 0]
    test_data = keras_data[test_index:, :input_timestep, :]
    test_label = keras_data[test_index:, input_timestep:, 0]
    return train_data, train_label, test_data, test_label


def train_data(data, name):
    data = np.array(data)
    scaler = sk.MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    scale = scaler.scale_[0]
    min = scaler.data_min_[0]

    train_data, train_label, test_data, test_label = \
        generate_keras_data(scaled_data, input_timestep, output_timestep)

    model = Sequential()
    model.add(CuDNNGRU(units=units, input_shape=(train_data.shape[1], train_data.shape[2]), return_sequences=True))
    model.add(Dropout(0.25))
    model.add(CuDNNGRU(units=units, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=train_label.shape[1]))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'mae'])
    model.summary()

    history = History()
    checkpoint = ModelCheckpoint('data/sentiment_analysis/results/'+name+'_{epoch:02d}.hdf5',
                                 monitor='val_acc', verbose=0, period=5)
    remote = RemoteMonitor()
    csv_log = CSVLogger('data/sentiment_analysis/results/'+name+'.log', separator=',')
    tensorboard = TensorBoard(log_dir='data/sentiment_analysis/results/'+name + '_logs', histogram_freq=2, write_grads=True)

    model.fit(train_data, train_label, batch_size=32, epochs=5, verbose=1, validation_split=0.1,
              callbacks=[history, checkpoint, remote, csv_log, tensorboard])

    print(history.history)
    with open('data/sentiment_analysis/results/history_'+name+'.pkl', 'wb') as hist_file:
        pickle.dump(history.history, hist_file)

    nn_pred = model.predict(test_data)
    true_label = (test_label / scale) + min
    pred_label = (nn_pred / scale) + min

    past_range = len(data)-len(pred_label)
    plt.plot(range(past_range-1000, past_range), data[past_range-1000:past_range,0], 'y', label = 'Past values')
    plt.plot(range(past_range, len(data)), true_label[:, 1], 'g', label = 'True values')
    plt.plot(range(past_range, len(data)), pred_label[:, 1], 'r', label = 'Predicted values')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc=0)
    plt.title('Prediction and true values')
    plt.savefig('data/sentiment_analysis/results/'+ name + '_input_' + str(input_timestep) + 'h')
    plt.show()
    plt.close()

    average_sign = []
    correct_sign = []
    mse = []
    mae = []

    for i in range(0, pred_label.shape[0]):
        past_label = test_data[i, :, :]
        past_label = scaler.inverse_transform(past_label)
        past_label = past_label[:, 0]

        sign_current = 0
        sign = 0
        for j in range(pred_label.shape[1]):
            sign_current = np.sign(pred_label[i, j] - past_label[-1])
            if sign_current == np.sign(true_label[i, j] - past_label[-1]):
                sign = 1
            else:
                sign = 0

        mse_current = np.mean((pred_label[i, 1] - true_label[i, 1]) ** 2)
        mse.append(mse_current)
        mae_current = np.absolute(pred_label[i, 1] - true_label[i, 1])
        mae.append(mae_current)
        average_sign.append(sign_current)
        correct_sign.append(sign)

    average_sign = np.mean(average_sign)
    correct_sign = np.mean(correct_sign)
    mse = np.mean(mse)
    mae = np.mean(mae)

    return average_sign, correct_sign, mse, mae

if __name__ == '__main__':
    # save_data()
    # merged_data = prepare_data()

    file = h5py.File('data/sentiment_analysis/results/bp_sent_02.hdf5', 'r')

    bp_sent_data = pd.read_pickle(save_path + 'merged_data.pkl')
    del bp_sent_data['date']
    bp_vad_data = bp_sent_data.drop(columns=['favourites', 'retweets', 'lex_sent', 'cnn_sent', 'cnn_pol'])
    bp_lex_data = bp_sent_data.drop(columns=['favourites', 'retweets', 'vad_sent', 'vad_pol', 'cnn_sent', 'cnn_pol'])
    bp_cnn_data = bp_sent_data.drop(columns=['favourites', 'retweets', 'lex_sent', 'vad_sent', 'vad_pol'])
    bp_data = bp_sent_data.drop(columns=['favourites', 'retweets', 'lex_sent', 'vad_sent', 'vad_pol', 'cnn_sent', 'cnn_pol'])
    sent_data = bp_sent_data[['favourites', 'retweets', 'lex_sent', 'vad_sent', 'vad_pol', 'cnn_sent', 'cnn_pol']]
    p_sent_data = bp_sent_data[['close', 'favourites', 'retweets', 'lex_sent', 'vad_sent', 'vad_pol', 'cnn_sent', 'cnn_pol']]

    average_sign_all, correct_sign_all, mse_all, mae_all = train_data(bp_sent_data, 'bp_sent')
    print('aaaa')
