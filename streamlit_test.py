import streamlit as st
import pandas as pd 
import pandas_datareader as web
import yfinance
import numpy as np
from datetime import datetime, date
from scipy.optimize import minimize

#Setting the Title of the Web app
st.write("""
# Mathematical Portfolio Optimizer
""")

#Getting all the ticker symbols from datahub.io
symbol = pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/0.csv")["Symbol"].values

#Creating a multiselect box to choose the tickers from
options = st.multiselect(
    'Enter Tickers of Companys',
    list(symbol),
    ['AAPL', 'FB']   
)

#Setting start and end date with the datetime module
start_date = st.date_input("Start Date", value = datetime.date(datetime(2018, 1, 1)))
end_date = st.date_input("End Date", value= datetime.today()) 

#Creating a empty Dataframe to store the data inside
df = pd.DataFrame()

end = datetime.today().strftime("%Y-%m-%d")

for o in options:
    ticker = yfinance.Ticker(o)
    df[o] = ticker.history(start = start_date, end = end)["Close"]

log_ret = np.log(df/df.shift(1))

#Creating some helper functions to calculate the expected return, expected volatility and Sharpe Ratio 

def get_ret_vol_st(weights):
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights)* 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))
    sr = ret/vol

    return np.array([ret,vol,sr])

def neg_sharpe(weights):
    return get_ret_vol_st(weights)[2] * -1

def check_sum(weights):
    return np.sum(weights) - 1


#Creating some variables to store some values inside which will be needed for the minimize function from scipy

cons = ({'type':'eq','fun':check_sum})

init_guess = [1/len(options) for i in options]

aaa = ((0,1),)
bounds = ()
for i in range(0,len(options)):
    bounds = bounds + aaa

#Calculating the optimal weights with the negative sharpe ratio
opt_results = minimize(neg_sharpe, init_guess, method='SLSQP',bounds = bounds, constraints=cons)


#Plotting all the data that we created and showing the optimal weights and the Portfolio stats
st.header("Closing Price of companys from Start Date - Today")
st.line_chart(df)

st.header("Data statistics")
st.write(df.describe())

st.header("Data Correlation")
st.write(df.pct_change().corr())

st.header("Optimal Portfolio Weights")
for i in range(len(options)):
    st.write(options[i], opt_results.x[i])

data = get_ret_vol_st(opt_results.x)

st.header("Optimal Portfolio stats")
st.write("Return: ", data[0])
st.write("Volatility: ",data[1])
st.write("Sharpe Ratio: ", data[2])
