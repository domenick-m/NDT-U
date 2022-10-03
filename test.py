from utils import (get_config)
from create_local_t5data import make_data


config = get_config({})

train_data, test_data = make_data(config)