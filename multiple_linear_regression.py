import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def model(btc_values_train, btc_values_test):

    # define the data/predictors as the pre-set feature names
    df = btc_values_train[:, 5:7]
    target = btc_values_train[:, 0]
    df_test = btc_values_test[:, 5:7]

    model = sm.OLS(target, df).fit()
    predictions = model.predict(df_test)
    summ = model.summary()
    print(summ)

    return predictions


btc_df = pd.read_pickle('data/tweets/poloniex/USDT_BTC.pkl')
btc_df_values = btc_df.values

all_data_size = len(btc_df_values)
btc_values_train = btc_df_values[:int(all_data_size * 0.9), :]
btc_values_test = btc_df_values[int(all_data_size * 0.9):, :]

pred_values = model(btc_values_train, btc_values_test)
truth_values = btc_values_test[:,0]

plt.figure(figsize=(8,6))
plt.plot(truth_values, label = 'Actual')
plt.plot(pred_values, 'r', label='Predicted')
plt.legend(loc='upper left')
plt.show()
print("aaaa")

