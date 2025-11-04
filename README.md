# Currency Value Forecasting
#In ten years, where will the cheapest Big Macs be?
#Ten year prediction of currency valuations relative to US Dollar
#This will be used to find undervalued currencies relative to US Dollar by using the Economist's Big Mac Index
#Utilizing ARIMA model since it is best suited for time series analysis
#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

#load dataset
file_path = "big-mac-adjusted-index.csv"
df = pd.read_csv(file_path)

#data quality check
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.nunique())

#Clean and prepare data
#convert date column and extract the year
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).copy()
df["year"] = df["date"].dt.year.astype(int)

#data quality check on cleaned data
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.nunique())

#countries with lowest Big Mac price averages
avg_prices = df.groupby('name')['adj_price'].mean()
lowest_countries = avg_prices.nsmallest(10).index
print("\n Top 10 undervalued Countries/Currencies relative to USD")
print(avg_prices.loc[lowest_countries])

#box plot of overall adjusted price
plt.figure(figsize=(8, 6))
df.boxplot(column='adj_price')
plt.title("Box Plot of Big Mac Adjusted Price (All Countries)")
plt.ylabel("Adjusted Price (USD)")
plt.tight_layout()
plt.show()

#Line plot by year
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['adj_price'], linestyle='-', marker='', alpha=0.6)
plt.title("Big Mac Adjusted Price Over Time (All Countries)")
plt.xlabel("Date")
plt.ylabel("Adjusted Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()

#filter for lowest 10 countries
df_lowest = df[df['name'].isin(lowest_countries)]

#forecast data function
forecast_data = {}

for country in lowest_countries:
    print(f"\n Forecasting for {country}")
    try:
        # Prepare country-level yearly data
        country_df = df_lowest[df_lowest['name'] == country].groupby('year')['adj_price'].mean()
        print(country_df)

        # Test stationarity
        result = adfuller(country_df)
        print(f" p-value: {result[1]:.4f} {'(Stationary)' if result[1] < 0.05 else '(Non-stationary)'}")

        # Train-test split
        train, test = train_test_split(country_df, test_size=0.2, shuffle=False)

        # Auto select ARIMA order
        model = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
        print(f" Best ARIMA order for {country}: {model.order}")

        # Forecast next 10 years
        model.fit(train)
        future_years = np.arange(country_df.index.max() + 1, country_df.index.max() + 11)
        forecast = model.predict(n_periods=10)

        # Store for plotting
        forecast_data[country] = (future_years, forecast)

        # plot residuals (residual = actual - predicted)
        residuals = pd.Series(model.resid())
        plt.figure(figsize=(10, 3))
        plt.plot(residuals)
        plt.title(f" Residuals for {country}")
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f" Error forecasting for (not enough data) {country}: {e}")
        forecast_data[country] = ([], [])

  #price forecast by country
plt.figure(figsize=(14, 7))
for country, (years, preds) in forecast_data.items():
    if len(years) > 0:
        plt.plot(years, preds, label=country)
plt.title("Predicted Big Mac Prices (Next 10 Years)\nTop 10 Undervalued Countries & Currencies")
plt.xlabel("Year")
plt.ylabel("Adjusted Price (USD)")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()



