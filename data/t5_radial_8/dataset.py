import sys
sys.path.append('/home/dmifsud/Projects/bg2-datasets/datasets')
from t5_radial_8 import init_toolkit_dataset as init_toolkit

def init_toolkit_dataset(session):
    return init_toolkit(session)