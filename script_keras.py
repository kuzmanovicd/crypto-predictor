import json
import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt

if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlopen


# connect to poloniex's API
CURRENCIES = ['USDT_BTC', 'USDT_LTC', 'USDT_ETH', 'USDT_XRP']
url = 'https://poloniex.com/public?command=returnChartData&currencyPair=$C&start=1356998100&end=9999999999&period=300'
urls = [url.replace('$C', c) for c in CURRENCIES]
#
# for i, c in enumerate(CURRENCIES):
#     with urlopen(urls[i]) as url:
#         r = url.read()
#         d = json.loads(r.decode())
#         df = pd.DataFrame(d)
#         #print(df.columns)
#         df.to_pickle('data/poloniex/' + c + '.pkl')
#         print('Successfully downloaded', c)

btc_df = pd.read_pickle('data/poloniex/USDT_BTC.pkl')


class PastSampler:

    def __init__(self, N, K, sliding_window=True):
        self.K = K
        self.N = N
        self.sliding_window = sliding_window

    def transform(self, A):
        M = self.N + self.K  # Number of samples per row (sample + target)
        # indexes
        if self.sliding_window:
            I = np.arange(M) + np.arange(A.shape[0] - M + 1).reshape(-1, 1)
        else:
            if A.shape[0] % M == 0:
                I = np.arange(M) + np.arange(0, A.shape[0], M).reshape(-1, 1)

            else:
                I = np.arange(M) + np.arange(0, A.shape[0] - M, M).reshape(-1, 1)

        print(I)
        print(I.shape)

        B = A[I].reshape(-1, M * A.shape[1], A.shape[2])
        ci = self.N * A.shape[1]  # Number of features per sample
        return B[:, :ci], B[:, ci:]  # Sample matrix, Target matrix


BTC_HEAD = btc_df.head()

import sklearn.preprocessing as prep
scaler = prep.MinMaxScaler()
#print(btc_df.as_matrix().shape)
#print(btc_df.as_matrix(), '#')
#print('#', np.array(scaler.fit_transform(btc_df))[:,None,:], '#')
#print(btc_df.shape)
original_A = np.array(btc_df)[:,None,:]
aa = np.array(btc_df)
A = np.array(scaler.fit_transform(btc_df))[:,None,:]
print(original_A.shape)
print(A.shape)
#print(prep.MinMaxScaler().fit_transform(A.reshape(-1,8)))
#print(A[1,0,:])
#print(scaler.inverse_transform(A.reshape(-1,8)), '#')
#print(scaler.get_params())


NPS, NFS = 256, 32         #Number of past and future samples
ps = PastSampler(NPS, NFS, sliding_window=True)
datas, labels = ps.transform(A)
print(datas.shape, labels.shape)

labels = labels[:,:,0].reshape(-1, NFS, 1)
print(labels.shape)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers import Conv1D, MaxPooling1D

from keras.layers import CuDNNLSTM, LSTM, LeakyReLU
from keras.callbacks import CSVLogger, ModelCheckpoint

step_size = datas.shape[1]
units = 50
second_units = 30
batch_size = NPS
nb_features = datas.shape[2]
epochs = 20
output_size = NFS
#split training validation
training_size = int(0.9 * datas.shape[0])
training_datas = datas[:training_size,:]
training_labels = labels[:training_size,:,0]
validation_datas = datas[training_size:,:]
validation_labels = labels[training_size:,:,0]


#build model
#if CuDNNLSTM is not working, use LSTM
model = Sequential()
model.add(CuDNNLSTM(units=units, input_shape=(step_size,nb_features),return_sequences=False))
model.add(Dropout(0.8))
model.add(Dense(output_size))
model.add(LeakyReLU())
model.compile(loss='mse', optimizer='adam')
model.summary()



model.fit(
    training_datas,
    training_labels,
    batch_size=batch_size,
    validation_split=0.2,
    #validation_data=(validation_datas,validation_labels),
    epochs=epochs,
    verbose=2,
    callbacks=[
        CSVLogger('nesto.csv', append=True)#,
        #ModelCheckpoint('nn_models/'+output_file_name+'-{epoch:02d}-{val_loss:.5f}.hdf5', verbose=1)
    ])

model.save('nn_models/btc_11_epochs.h5')

original_A.reshape(-1,8)[:,0].shape

scaler2 = prep.MinMaxScaler()
scaler2.fit_transform(original_A.reshape(-1,8)[:,0].reshape(-1,1))
vd = validation_datas[0:len(validation_datas):NPS,:,:]

predicted = model.predict(vd)

vd = scaler.inverse_transform(vd.reshape(-1,8))


predict = predicted.reshape(-1,1)
truth = validation_labels[0:len(validation_labels):NPS,:].reshape(-1,1)

predict = scaler2.inverse_transform(predict)
truth = scaler2.inverse_transform(truth)

plt.figure(figsize=(8,6))
plt.plot(truth, label = 'Actual')
plt.plot(predict, 'r', label='Predicted')
plt.legend(loc='upper left')
plt.show()