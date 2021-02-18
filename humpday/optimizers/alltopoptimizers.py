from humpday.optimizers.pysotcube import pysot_srbf_cube, pysot_dycors_cube
from humpday.optimizers.axcube import ax_default_cube
from humpday.optimizers.dlibcube import dlib_default_cube
from humpday.optimizers.nevergradcube import nevergrad_ngopt8_cube, nevergrad_ngopt4_cube
from humpday.optimizers.optunacube import optuna_cmaes_cube
from humpday.optimizers.pymoocube import pymoo_pattern_cube
from humpday.optimizers.shgocube import shgo_powell_sobol_cube
from humpday.optimizers.skoptcube import skopt_gp_default_cube
from humpday.optimizers.nloptcube import nlopt_directr_cube

TOP_OPTIMIZERS = [pysot_dycors_cube, pysot_srbf_cube, ax_default_cube, dlib_default_cube,
                  nevergrad_ngopt4_cube, nevergrad_ngopt8_cube, optuna_cmaes_cube, pymoo_pattern_cube,
                  shgo_powell_sobol_cube, skopt_gp_default_cube, nlopt_directr_cube]
