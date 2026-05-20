"""
Thurstone-based rating system to replace the ELO implementation.
Uses proper statistical modeling for optimizer comparisons.
"""

import numpy as np
import random
from typing import Dict, List, Any, Optional
from pprint import pprint
import traceback

from thurstone import GlobalAbilityCalibrator
from humpday.objectives.classic import CLASSIC_OBJECTIVES
from humpday.optimizers.alloptimizers import OPTIMIZERS

N_DIM_CHOICES = [1, 2, 3, 5, 8]
N_TRIALS_CHOICES = [130, 210, 340]


class OptimizerThurstoneCaliber:
    """Thurstone-based calibration for optimizer performance comparisons."""

    def __init__(self, initial_theta: float = 0.0):
        self.calibrator = None
        self.optimizer_ids = []
        self.race_history = []
        self.initial_theta = initial_theta

    def initialize_optimizers(self, optimizers: List[Any]):
        """Initialize the calibrator with a list of optimizers."""
        self.optimizer_ids = [opt.__name__ for opt in optimizers]
        self.calibrator = GlobalAbilityCalibrator(
            horse_ids=self.optimizer_ids,
            theta={opt_id: self.initial_theta for opt_id in self.optimizer_ids}
        )

    def add_comparison_result(self, game_result: Dict[str, Any]):
        """Add a comparison result to update the ratings."""
        if not game_result.get('completed', False):
            return

        white_name = game_result['white'].__name__
        black_name = game_result['black'].__name__
        points = game_result['points']

        # Convert points to finish positions (0=first, 1=second for binary race)
        if points > 0.75:  # White wins
            finish = [white_name, black_name]
        elif points < 0.25:  # Black wins
            finish = [black_name, white_name]
        else:  # Draw - randomly assign first/second
            finish = [white_name, black_name] if random.random() > 0.5 else [black_name, white_name]

        race_spec = {
            'finish': finish,
            'n_dim': game_result['n_dim'],
            'n_trials': game_result['n_trials'],
            'objective': game_result['objective']
        }

        self.race_history.append(race_spec)
        self.calibrator.add_race(finish)

    def fit(self) -> bool:
        """Fit the model to update ability estimates."""
        if self.calibrator is None or len(self.race_history) < 2:
            return False
        try:
            self.calibrator.fit()
            return True
        except Exception as e:
            print(f"Error fitting calibrator: {e}")
            return False

    def get_ratings(self) -> Dict[str, float]:
        """Get current ability ratings for all optimizers."""
        if self.calibrator is None:
            return {}
        return dict(self.calibrator.theta)

    def get_leaderboard(self) -> List[tuple]:
        """Get sorted leaderboard of (rating, name) pairs."""
        ratings = self.get_ratings()
        return sorted([(rating, name) for name, rating in ratings.items()], reverse=True)


def optimizer_game_thurstone(white, black, n_dim, n_trials, objective, tol=0.001):
    """
    Same interface as the original optimizer_game but optimized for Thurstone rating.
    Returns the same game_result format for compatibility.
    """
    # Import the original function for now to maintain compatibility
    from humpday.comparison.eloratings import optimizer_game
    return optimizer_game(white, black, n_dim, n_trials, objective, tol)


def random_optimizer_game_thurstone(optimizers=None, objectives=None, n_dim_choices=None,
                                  n_trials_choices=None, tol=0.001, announce=False, pattern=None):
    """Thurstone version of random optimizer game."""
    if n_dim_choices is None:
        n_dim_choices = N_DIM_CHOICES
    if n_trials_choices is None:
        n_trials_choices = N_TRIALS_CHOICES
    if objectives is None:
        objectives = CLASSIC_OBJECTIVES
    if optimizers is None:
        optimizers = OPTIMIZERS

    n_attempts_left = 1000
    found = False
    while n_attempts_left > 0 and not found:
        n_attempts_left -= 1
        white, black = np.random.choice(optimizers, size=2, replace=False)
        if pattern is None or (pattern in white.__name__) or (pattern in black.__name__):
            found = True

    if not found:
        pprint(optimizers)
        raise ValueError('No optimizer matches ' + pattern)

    matchup = {
        'n_dim': random.choice(n_dim_choices),
        'n_trials': random.choice(n_trials_choices),
        'white': white,
        'black': black,
        'objective': random.choice(objectives),
        'tol': tol
    }
    if announce:
        pprint(matchup)
    return optimizer_game_thurstone(**matchup)


def demo_thurstone_ratings():
    """Demo function for running Thurstone-based optimizer ratings."""
    caliber = OptimizerThurstoneCaliber()
    caliber.initialize_optimizers(OPTIMIZERS)

    game_count = 0
    while True:
        game_result = random_optimizer_game_thurstone(
            optimizers=OPTIMIZERS,
            objectives=CLASSIC_OBJECTIVES,
            n_dim_choices=N_DIM_CHOICES,
            n_trials_choices=N_TRIALS_CHOICES,
            tol=0.001
        )

        print(f' Game {game_count + 1}...')
        pprint(game_result)

        caliber.add_comparison_result(game_result)
        game_count += 1

        # Fit every 5 games
        if game_count % 5 == 0:
            if caliber.fit():
                print(f'\n After {game_count} games:')
                leaderboard = caliber.get_leaderboard()
                for rating, name in leaderboard[:10]:  # Top 10
                    print(f"  {rating:.2f}: {name}")
                print()


if __name__ == '__main__':
    demo_thurstone_ratings()