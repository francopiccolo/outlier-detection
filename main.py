# tickers without duplication?

from datetime import date
import string

from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from pandas import DatetimeIndex

from config import DB_PATH, NUM_STOCKS, TABLE_NAME, START_DATE, END_DATE

def generate_price_data(
    n: int, m: int, mu: float = 0.08, sigma: float = 0.15
    ):
    """
    Generate an array of plausible random stock prices histories.
    Each row of the array represents a different stock ticker.
    Each column of the array represents a different date.
    1 out of 10000 values is multiplied by a random number between 0 and 2
    The first price in each series is untouched by randomness
    :param n: number of stocks
    :param m: number of dates
    :param mu: annualised drift of the stock
    :param sigma: annualised volatility of the stock
    :return: n x m numpy array
    """
    dt = 1. / 255
    dW = np.random.normal(0, sigma * np.sqrt(dt), size=(n, m-1))
    S0 = np.random.uniform(10, 1000, size=(n, 1))
    mults = 1 + mu * dt + dW
    prices = np.cumprod(np.hstack((S0, mults)), axis=1)
    random_multiplier = create_random_multiplier_array(n, m, 0.0001)
    random_multiplier[:, 0] = 1
    prices = np.multiply(prices, random_multiplier)
    return np.round(prices, decimals=2)

def create_random_multiplier_array(n: int, m: int, p: float):
    """
    Generate an array whose elements are with a probability of p a random number
    between 0 and 2 and with 1 - p probability 1.

    :param n: number of rows
    :param m: number of columns
    :param p: probability of assigning a random number

    :return: n x m numpy array
    """
    random_numbers = np.random.uniform(low=0, high=2, size=(n, m))
    random_activation = np.random.choice([1, 0], p=[p, 1 - p], size=(n, m))
    random_multiplier = np.where(random_activation == 1, random_numbers, 1)
    return random_multiplier

def generate_date_array(start_date: date, end_date: date) -> DatetimeIndex:
    """
    Generates a DatetimeIndex from start_date to end_date excluding weekends
    :param start_date: Start date
    :param end_date: End date
    :return: DatetimeIndex
    """
    return pd.date_range(start_date, end_date, freq='B')

def generate_tickers(n: int) -> list:
    """
    Generates a list with n distinct tickers each formed by 4 random not duplicated uppercase letters
    :param n: Number of tickers
    :param end_date: End date
    :return: list of size n
    """
    tickers = set()
    while len(tickers) < n:
        ticker = np.random.choice(np.array(list(string.ascii_uppercase)), 4, replace=False)
        tickers.add(''.join(ticker))
    
    return list(tickers)

def create_test_data(start_date: date, end_date: date, num_stocks: int) -> pd.DataFrame:
    """
    Creates a dataframe with test data between start_date and end_date for num_stocks number of stocks.
    :param start_date: Start date
    :param end_date: End date
    :num_stocks: Number of stocks
    :return: pd.DataFrame
    """
    dates_array = generate_date_array(start_date=start_date, end_date=end_date)
    tickers = generate_tickers(n=num_stocks)
    data = generate_price_data(n=num_stocks, m=len(dates_array))
    return pd.DataFrame(data=data.T, index=dates_array, columns=tickers)

def load_data_to_sqlite(df: pd.DataFrame, table: str, db_path: str) -> None:
    """
    Loads the dataframe df to sqlite table table.
    :param df: Data dataframe
    :param table: SQLite table name
    :param db_path: SQLite db path
    :return: None
    """
    
    engine = create_engine(db_path)
    df = df.reset_index()
    df['index'] = df['index'].dt.strftime('%Y-%m-%d')
    df.to_sql(name=table, con=engine, if_exists='replace', index=False)


def get_data_from_sqlite(table: str, db_path: str) -> pd.DataFrame:
    """
    Gets data from sqlite table table to a dataframe
    :param table: SQLite table name
    :param db_path: SQLite db path
    :return: DataFrame
    """
    engine = create_engine(db_path)
    df = pd.read_sql_table(table_name=table, con=engine)
    df['index'] = pd.to_datetime(df['index'])
    df = df.set_index('index')
    return df

def locate_and_replace_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Locates outliers in the dataframe where the stock price has jumped (e.g. more than 10%) vs
the surrounding values.
    After that replaces any outliers by the previous valid value in each time series.
    Also prints out the number of outliers found, bucketed by the first letter of the ticker.
    :param df: Stocks data dataframe
    :return: Dataframe with data without outliers
    """
    pass

    

def main():
    df = create_test_data(start_date=START_DATE, end_date=END_DATE, num_stocks=NUM_STOCKS)
    load_data_to_sqlite(df=df, table=TABLE_NAME, db_path=DB_PATH)
    df = get_data_from_sqlite(table=TABLE_NAME, db_path=DB_PATH)
    df = locate_and_replace_outliers(df=df)
    load_data_to_sqlite(df=df, table=TABLE_NAME)    
    # test locate_and_replace_outliers
    

if __name__ == '__main__':
    df = create_test_data(start_date=START_DATE, end_date=END_DATE, num_stocks=NUM_STOCKS)
    load_data_to_sqlite(df=df, table=TABLE_NAME, db_path=DB_PATH)
    df = get_data_from_sqlite(table=TABLE_NAME, db_path=DB_PATH)
    