from datetime import datetime

import pandas as pd

from main import locate_and_replace_outliers_in_col

def test_locate_and_replace_outliers_in_col() -> None:
    """
    Tests that the function locate_and_replace_outliers_in_col works as expected
    by creating a sample series, applying the function and comparing the result with the expected result    
    :return: None
    """
    sample_series = pd.Series(
        data=[102.76, 103.2, 185.44, 102.81, 106.45, 101.8],
        index=[datetime(2023, 4, 3), datetime(2023, 4, 4), datetime(2023, 4, 5), datetime(2023, 4, 6), datetime(2023, 4, 7), datetime(2023, 4, 10)],
        name='data'
    )

    expected_series = pd.Series(
        data=[102.76, 103.2, 103.2, 102.81, 106.45, 101.8],
        index=[datetime(2023, 4, 3), datetime(2023, 4, 4), datetime(2023, 4, 5), datetime(2023, 4, 6), datetime(2023, 4, 7), datetime(2023, 4, 10)],
        name='data'
    )

    expected_num_outliers = 1

    result_series, result_num_outliers = locate_and_replace_outliers_in_col(sample_series)

    assert (result_series == expected_series).all()
    assert expected_num_outliers == result_num_outliers