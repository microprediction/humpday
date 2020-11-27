import numpy as np
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from microprediction import MicroReader

mr = MicroReader()
lagged_values = mr.get_lagged_values(name='electricity-fueltype-nyiso-dual_fuel.json')












ARmodel = ar_select_order(lagged_values, maxlag=6)
model_fit = ARmodel.model.fit()
point_est = model_fit.predict(start=len(lagged_values), end=len(lagged_values), dynamic=False)