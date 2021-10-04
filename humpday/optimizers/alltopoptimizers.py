# So horrendous this needs a rewrite

try:
    from humpday.optimizers.nevergradcube import nevergrad_ngopt8_cube, nevergrad_ngopt4_cube
    from humpday.optimizers.optunacube import optuna_cmaes_cube
    from humpday.optimizers.shgocube import shgo_powell_sobol_cube
    from humpday.optimizers.pymoocube import pymoo_pattern_cube
    from humpday.optimizers.nloptcube import nlopt_directr_cube
    from humpday.optimizers.pysotcube import pysot_srbf_cube, pysot_dycors_cube
    from humpday.optimizers.dlibcube import DLIB_OPTIMIZERS
    TOP_OPTIMIZERS = [pysot_dycors_cube, pysot_srbf_cube,
                      nevergrad_ngopt4_cube, nevergrad_ngopt8_cube, optuna_cmaes_cube, pymoo_pattern_cube,
                      shgo_powell_sobol_cube, nlopt_directr_cube] + DLIB_OPTIMIZERS
except ImportError:
    print('To use top optimizers in anger you need to install pysot, nevergrad, optuna, pymoo, skopt, nlopt, dlib')
    from humpday.optimizers.nevergradcube import nevergrad_ngopt8_cube, nevergrad_ngopt4_cube
    from humpday.optimizers.shgocube import shgo_powell_sobol_cube
    from humpday.optimizers.optunacube import optuna_cmaes_cube

    TOP_OPTIMIZERS = [nevergrad_ngopt4_cube, nevergrad_ngopt8_cube, optuna_cmaes_cube,shgo_powell_sobol_cube]






