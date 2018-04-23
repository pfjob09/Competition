import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet

df = pd.read_csv('data/train.csv')

df.drop(['id', 'questions'], axis=1, inplace=True)

df.rename(columns={'answers': 'y', 'date': 'ds'}, inplace=True)

print(df.head())

df['y'] = np.log(df['y'])

print(df.head())

m = Prophet()

m.fit(df)

future = m.make_future_dataframe(periods=152)

print(future.tail())

forecast = m.predict(future)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

m.plot(forecast)

plt.show()

m.plot_components(forecast)

plt.show()

submit = pd.read_csv('data/sample_submit.csv')

submit['answers'] = np.exp(forecast['yhat'])

submit.to_csv('result.csv', index=None)
"""
df.drop(['id', 'answers'], axis=1, inplace=True)

df.rename(columns={'questions': 'y', 'date': 'ds'}, inplace=True)

print(df.head())

df['y'] = np.log(df['y'])

print(df.head())

m = Prophet()

m.fit(df)

future = m.make_future_dataframe(periods=152)

print(future.tail())

forecast = m.predict(future)

# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# m.plot(forecast)

# plt.show()

submit = pd.read_csv('result.csv')

submit['questions'] = np.exp(forecast['yhat'])

submit.to_csv('result.csv', index=None)
"""