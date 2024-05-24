from datetime import datetime

DB_PATH = 'sqlite:///data.sqlite'
TABLE_NAME = 'stocks'
NUM_STOCKS = 5
START_DATE = datetime(2024, 1, 1).date()
END_DATE = datetime(2024, 1, 10).date()
RANDOMIZATION_PROB = 0.2