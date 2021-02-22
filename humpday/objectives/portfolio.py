

# Objectives inspired by Markowitz

import time
import numpy as np
from sklearn.datasets import make_spd_matrix
from datetime import datetime
day_of_year = datetime.now().timetuple().tm_yday
SIGMA = make_spd_matrix(n_dim=500, random_state=day_of_year)


def risk_on_cube(u:[float])->float:
    """
    :param u:    Proportion of portfolio in first n
    :return:
    """
    x_dim = len(u)+1
    x = list(u)+ [1-np.sum(u)]
    sigma_matrix = SIGMA[0:x_dim,0:x_dim]
    portfolio_var = np.linalg.multi_dot( [np.array(x).transpose(), sigma_matrix, x] )
    return portfolio_var



PORTFOLIO_OBJECTIVES = [ risk_on_cube ]



if __name__=="__main__":
    for objective in PORTFOLIO_OBJECTIVES:
        objective(u=[0.0,0.5,1.0])
        objective(u=[0.0, 0.5, 0.0, 0.0, 1.0])
    print(len(PORTFOLIO_OBJECTIVES))




