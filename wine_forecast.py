#########################################################
#
# Author: Timur Dzhafari
#
# o	Linear trend
# o	Seasonality
# o	Linear trend and seasonality
# o	Simple exponential smoothing model
#
#########################################################

import pandas as pd
import matplotlib.pylab as plt
import statsmodels.formula.api as sm
import numpy as np
from statsmodels.tsa import tsatools
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from dmba import regressionSummary

SHOWVIZ = True

df_td = pd.read_csv(r"D:\School\UNCC\projects\repos\wine_forecast\AustralianWines.csv")

# formatting and cleaning
df_td["Date"] = pd.to_datetime(df_td.Month, format="%b-%y")

for col in df_td.columns:
    if col[:3] == "Red":
        name_to_change = col

df_td.rename({name_to_change: "Red"}, axis="columns", inplace=True)
df_td.dropna(inplace=True)
df_td["Red"] = pd.to_numeric(df_td["Red"], errors="coerce").astype("int")

wine_ts_td = pd.Series(df_td["Red"].values, index=df_td.Date, name="Red")

print(df_td.describe())

# print(wine_ts_td)

# visualization
if SHOWVIZ == True:
    ax = wine_ts_td.plot()
    ax.set_xlabel("Time")
    ax.set_ylabel("Red")
    ax.set_ylim(400, 3800)
    plt.show()

result=seasonal_decompose(df_td['Red'], model='additive', period=12)
result.plot()

ts_df_td = tsatools.add_trend(wine_ts_td, trend='t')
ts_df_td['Month'] = ts_df_td.index.month

nTrain = ts_df_td.shape[1] - 24

wine_train_ts = wine_ts_td[:nTrain] 
wine_valid_ts = wine_ts_td[nTrain:]

train_df_td = ts_df_td[:nTrain]
valid_df_td = ts_df_td[nTrain:]

wine_lm_td = sm.ols(formula='Red~trend',data=train_df_td).fit()

print('Linear trend. Model summary and errors')
print(wine_lm_td.summary())

predict_lm_td = wine_lm_td.predict(valid_df_td)

regressionSummary(wine_valid_ts, predict_lm_td)

# Seasonality

wine_lm_season_td = sm.ols(formula='Red~C(Month)',data=train_df_td).fit()
print('Seasonality. Model summary and errors')
print(wine_lm_season_td.summary())
predict_lm_season =wine_lm_season_td.predict(valid_df_td)
regressionSummary(wine_valid_ts, predict_lm_season)

# Linear trend + seasonality 

modelfomula = 'Red~trend+np.square(trend)+C(Month)'
wine_lm_trendseason = sm.ols(formula=modelfomula,
                              data=train_df_td).fit()

print('Seasonality and trend. Model summary and errors')
print(wine_lm_trendseason.summary())
predict_lm_trendseason = wine_lm_trendseason.predict(valid_df_td)
regressionSummary(wine_valid_ts, predict_lm_trendseason)

# Simple exponential smoothing

SES = SimpleExpSmoothing(wine_train_ts,
                         initialization_method='estimated').fit() # estimated method is used, which estimates the initial smoothing level based on the data; which finds the value of alpha that minimizes the sum of squared errors between the predicted and actual values
predict_SES = SES.forecast(len(wine_valid_ts))

SES.model.params

regressionSummary(wine_valid_ts, predict_SES)
print('Simple exponential smoothing. Model summary and errors')
print(SES.summary())
combine_SES = pd.concat([SES.fittedvalues, predict_SES])

if SHOWVIZ:
    plt.figure(figsize=(20,5))
    plt.grid()
    plt.plot(wine_train_ts, label='Train')
    plt.plot(wine_valid_ts, label='Validation')
    plt.plot(combine_SES, label='Exponential Smoothing Forecast')
    plt.legend(loc='best')
    plt.title('Exponential Smoothing Forecast')
    plt.show()