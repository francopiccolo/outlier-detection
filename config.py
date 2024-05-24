from datetime import datetime

DB_PATH = 'sqlite:///data.sqlite'
TABLE_NAME = 'stocks'
NUM_STOCKS = 10000
START_DATE = datetime(2020, 1, 1).date()
END_DATE = datetime(2023, 12, 29).date()
RANDOMIZATION_PROB = 0.0001