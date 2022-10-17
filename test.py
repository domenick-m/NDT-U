from utils.data.create_local_t5data import get_trial_data
from utils_f import get_config
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
from utils.plot.plot_true_vs_pred_mvmnt import plot_true_vs_pred_mvmnt


def test():

    # take in model checkpoint

    # loop through data and run model on trials, need to get this from datasets

    # plot each session pre-readout outputs colored by condition
    pass
