from getjson import getjson
from pprint import pprint

TIMING_URL='https://raw.githubusercontent.com/microprediction/humpday/main/humpday/comparison/timing.json'


def get_timing():
    return getjson(TIMING_URL)


def get_strategy_time(optimizer,n_dim:int=5, n_trials:int=130):
    """ Return approximate time an optimizer will run for, from past trials
    :param optimizer:   str or optimizer
    :param n_dim:
    :param n_trials:
    :return:
    """
    if isinstance(optimizer,str):
        o = optimizer if '_cube' in optimizer else optimizer+'_cube'
    else:
        o = optimizer.__name__
    cpu = get_timing()
    try:
        return cpu[o][n_dim][n_trials]
    except Exception:
        return None


if __name__=='__main__':
    pprint(get_timing())