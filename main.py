from datetime import date
import string

from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from pandas import DatetimeIndex

from config import DB_PATH, NUM_STOCKS, TABLE_NAME, START_DATE, END_DATE, RANDOMIZATION_PROB

def generate_price_data(
    n: int, m: int, mu: float = 0.08, sigma: float = 0.15, randomization_prob: float = 0.0001
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
    random_multiplier = create_random_multiplier_array(n, m, randomization_prob)
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
    print(random_multiplier.T)
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

def create_test_data(start_date: date, end_date: date, num_stocks: int, randomization_prob: float ) -> pd.DataFrame:
    """
    Creates a dataframe with test data between start_date and end_date for num_stocks number of stocks.
    :param start_date: Start date
    :param end_date: End date
    :num_stocks: Number of stocks
    :randomization_prob: Probability of randomizing a value from the test data
    :return: pd.DataFrame
    """
    dates_array = generate_date_array(start_date=start_date, end_date=end_date)
    tickers = generate_tickers(n=num_stocks)
    data = generate_price_data(n=num_stocks, m=len(dates_array), randomization_prob=randomization_prob)
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

def locate_and_replace_outliers_in_col(series: pd.Series) -> (pd.Series, int):
    """
    Locates outliers in a series stock price has jumped (e.g. more than 10%) vs
the surrounding values.
    After that replaces any outliers by the previous valid value the time series.
    Also prints out the number of outliers found, bucketed by the first letter of the ticker.
    :param series: Series of stock prices for a ticker
    :return: Series of stock prices with their outliers removed, number of outliers removed
    """
    num_outliers = 0
    col_name = series.name
    df = pd.DataFrame(series)
    df['prev_value'] = df[col_name].shift()
    df['abs_diff_with_prev_value'] = abs(df[col_name] - df['prev_value'])
    df['pct_diff_with_prev_value'] = df['abs_diff_with_prev_value'] / df[col_name]
    df['abs_diff_with_next_value'] = abs(df[col_name] - df[col_name].shift(-1))
    df['pct_diff_with_next_value'] = df['abs_diff_with_next_value'] / df[col_name]
    df['new_val'] = df.apply(lambda row: row['prev_value'] if row['pct_diff_with_prev_value'] > 0.1 and row['pct_diff_with_next_value'] > 0.1 else row[col_name], axis=1)
    df['is_outlier'] = df[col_name] != df['new_val']
    num_outliers = sum(df['is_outlier'])

    return series, num_outliers


def locate_and_replace_outliers_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Locates outliers in the dataframe where the stock price has jumped (e.g. more than 10%) vs
the surrounding values.
    After that replaces any outliers by the previous valid value in each time series.
    Also prints out the number of outliers found, bucketed by the first letter of the ticker.
    :param df: Stocks data dataframe
    :return: Dataframe with data without outliers
    """
    outliers = {}
    for col in df.columns:
        new_col, num_outliers = locate_and_replace_outliers_in_col(df[col])
        df[col] = new_col
        if col[0] not in outliers:
            outliers[col[0]] = num_outliers
        else:
            outliers[col[0]] += num_outliers
    
    print('Number of outliers found, bucketed by the first letter of the ticker \n', outliers)
    return df
    

def pipeline(start_date: date, end_date: date, num_stocks: int, db_path: str, table_name: str, randomization_prob: float) -> None:
    """
    Executes the whole pipeline, from test data generation to outlier detection
    :param start_date: Start date
    :param end_date: End date
    :param num_stocks: Number of stocks
    :param db_path: SQLite path
    :param table_name: SQLite table name
    :param randomization_prob: Probability of randomizing the test data
    :return: None
    """

    df = create_test_data(start_date=start_date, end_date=end_date, num_stocks=num_stocks, randomization_prob=randomization_prob)
    load_data_to_sqlite(df=df, table=table_name, db_path=db_path)
    df = get_data_from_sqlite(table=table_name, db_path=db_path)
    df = locate_and_replace_outliers_in_df(df=df)
    load_data_to_sqlite(df=df, table=table_name, db_path=db_path)
    # test locate_and_replace_outliers
    

if __name__ == '__main__':
    pipeline(start_date=START_DATE, end_date=END_DATE, num_stocks=NUM_STOCKS, db_path=DB_PATH, table_name=TABLE_NAME, randomization_prob=RANDOMIZATION_PROB)    