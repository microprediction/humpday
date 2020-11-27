# Quick test of install

import optuna
from optuna.logging import CRITICAL

def test_optuna():

    def objective(trial):
        x = trial.suggest_uniform('x', -10, 10)
        return (x - 2) ** 2

    optuna.logging.set_verbosity(CRITICAL)
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)