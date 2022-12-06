import sys
sys.path.append('/home/dmifsud/Projects/BG2-Datasets/datasets')
from t11_piano import init_toolkit_dataset as init_toolkit

def init_toolkit_dataset(session):
    return init_toolkit(session)