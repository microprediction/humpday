
from humpday.optimizers.adaptive_optimizer import (
    EloRatingSystem,
    pure_optimize,
    sphere_variants_generator,
)


def test_elo_optim():
    """Test the new EloRatingSystem functionality."""
    # Test basic Elo system functionality
    elo = EloRatingSystem()

    # Test that all algorithms start with default rating
    initial_rating = elo.get_rating("NelderMead")
    assert initial_rating == 1500.0

    # Test rating updates
    elo.update_ratings("NelderMead", "RandomSearch", 1.0)  # NelderMead wins
    assert elo.get_rating("NelderMead") > 1500.0  # Should increase
    assert elo.get_rating("RandomSearch") < 1500.0  # Should decrease

    # Test with objective generator
    generator = sphere_variants_generator(n_dim=3)
    objective = next(generator)

    # Test that pure_optimize works with different algorithms
    result1 = pure_optimize(objective, "NelderMead", n_trials=8, n_dim=3)
    result2 = pure_optimize(objective, "RandomSearch", n_trials=8, n_dim=3)

    assert len(result1) == 2  # (best_value, best_x)
    assert len(result2) == 2


if __name__ == "__main__":
    test_elo_optim()
