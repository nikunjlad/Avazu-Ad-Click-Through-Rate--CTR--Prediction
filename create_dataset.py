
import pandas as pd
import random
import numpy as np
from datetime import datetime

num_records = 40428967

sample_size = 200000
skip_values = sorted(random.sample(range(1,num_records), num_records - sample_size))

types_train = {
    'id': np.dtype(int),
    'click': np.dtype(int),
    'hour': np.dtype(datetime),
    'C1': np.dtype(int),
    'banner_pos': np.dtype(int),
    'site_id': np.dtype(str),
    'site_domain': np.dtype(str),
    'site_category': np.dtype(str),
    'app_id': np.dtype(str),
    'app_domain': np.dtype(str),
    'app_category': np.dtype(str),
    'device_id': np.dtype(str),
    'device_ip': np.dtype(str),
    'device_model': np.dtype(str),
    'device_type': np.dtype(int),
    'device_conn_type': np.dtype(int),
    'C14': np.dtype(int),
    'C15': np.dtype(int),
    'C16': np.dtype(int),
    'C17': np.dtype(int),
    'C18': np.dtype(int),
    'C19': np.dtype(int),
    'C20': np.dtype(int),
    'C21':np.dtype(int)
}

parse_date = lambda val : datetime.strptime(val, '%y%m%d%H')

reduced = pd.read_csv("data/train.csv", parse_dates=['hour'], date_parser=parse_date, dtype=types_train, skiprows=skip_values)
reduced.to_csv("data/train_subset.csv",index=False)