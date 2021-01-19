import numpy as np
import datetime
import scipy.optimize as optimization
import pandas as pd
# pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as web
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

stocks = ['AAPL', 'TSLA', 'DB', 'JPM', 'WMT', 'GE', 'AMZN']

start_date = '01/01/2020'
end_date = '18/01/2021'

# downloading data from yahoo finance


def data_download(stocks):
    data = web.DataReader(stocks, data_source='yahoo', start=start_date, end=end_date)['Adj Close']
    data.columns = stocks
    return data


def data_plot(data):
    data.plot(figsize=(15, 5))
    plt.show()

# returns calculation from data and ploting


def calculate_returns(data):
    returns = np.log(data/data.shift(1))
    return returns


def plot_returns(returns):
    returns.plot(figsize=(15, 10))
    plt.show()

# print out mean nd co-variance


def covmean(returns):
    print(returns.mean()*252)
    print(returns.cov()*252)


def stock_weights():
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    return weights


def port_returns(returns, weights):
    portfolio_return = np.sum(returns.mean()*weights)*252
    print("Expected portfolio return for listed stocks is : ", portfolio_return)


def port_variance(returns, weights):
    portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))
    print("Expected portfolio variance from listed stocks is : ", portfolio_variance)


# Monte carlo simulation to generate portfolios

def new_portfolios(weights, returns):
    preturns = []
    pvariance = []

    for i in range(5000):
        weights = np.random.random(len(stocks))
        weights /= np.sum(weights)
        preturns.append(np.sum(returns.mean(), *weights)*252)
        pvariance.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights))))

    preturns = np.array(preturns)
    pvariance = np.array(pvariance)
    return pvariance, preturns


def plot_portfolios(returns, variances):
    plt.figure(figsize=(20, 15))
    plt.scatter(variances, returns, c=returns/variances, marker='o')
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Returns")
    plt.colorbar(label="Sharpe Ratio")
    plt.show()

# now finding the best possible portfolio using optimization


def statistics(weights, returns):
    portfolio_returns = np.sum(returns.mean()*weights.T)*252
    portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))
    return np.array([portfolio_returns, portfolio_variance, portfolio_returns/portfolio_variance])


def sharpe_min_func(weights, returns):
    return -statistics(weights, returns)[2]


def optimal_portfolio(weights, returns):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
    bounds = tuple((0, 1) for x in range(len(stocks)))
    optimal = optimization.minimize(fun=sharpe_min_func, x0=weights, args=returns, method='SLSQP',
                                    bounds=bounds, constraints=constraints)
    return optimal

# print optimal portfolio


def print_optimal_port(optimal, returns):
    print("Optimal weights", optimal['x'].round(3))
    print("Expected returns, volatility & sharpe ratio: ", statistics(optimal['x'].round(3), returns))


def optimal_port_plot(pvariance, preturns, optimal, returns):
    plt.figure(figsize=(20, 15))
    plt.scatter(pvariance, preturns, c=preturns/pvariance, marker='o')
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Returns")
    plt.colorbar(label="Sharpe Ratio")
    plt.plot(statistics(optimal['x'], returns[1]), statistics(optimal['x'], returns[0]), 'g*', markersize=20.0)
    plt.show()


if __name__ == '__main__':
    data = data_download(stocks)
    data_plot(data)
    returns = calculate_returns(data)
    calculate_returns(returns)
    weights = stock_weights()
    port_returns(returns, weights)
    port_variance(returns, weights)
    vals = new_portfolios(weights, returns)
    plot_portfolios(returns, variances=1)
    preturns = vals[0]
    pvariances = vals[1]
    statistics(weights, returns)
    optimal = optimal_portfolio(weights, returns)
    new_portfolios(weights, returns)
    optimal_portfolio(returns=0, weights=1)
    optimal_port_plot(preturns, pvariances, optimal, returns)
    print_optimal_port(optimal, returns, preturns, pvariances)
