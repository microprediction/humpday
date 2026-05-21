# Adaptive Optimization with Elo Rating System

Humpday now includes an advanced adaptive optimization system that learns which algorithms work best for different types of problems. Instead of guessing which algorithm to use, the system runs algorithm tournaments, maintains Elo ratings based on performance, and provides data-driven recommendations.

## Key Features

- **Automatic Algorithm Selection**: System learns which algorithms work best through testing
- **Elo Rating System**: Maintains skill ratings for all 22 algorithms based on head-to-head performance
- **Objective Generator Interface**: Takes functions that generate diverse test problems
- **Budget-Aware Testing**: Efficiently allocates evaluation budget between exploration and exploitation
- **Adaptive Recommendations**: Provides suggestions tailored to problem characteristics

## Quick Start

```python
from humpday import adaptive_optimize, sphere_variants_generator

# Create an objective generator (yields test problems)
objective_gen = sphere_variants_generator(n_dim=3)

# Run adaptive optimization
results = adaptive_optimize(
    objective_generator=objective_gen,
    trials_budget=2000,
    n_dim=3,
    n_warmup_problems=5,
    trials_per_warmup=50
)

# Get the best algorithms
top_algorithms = results['top_algorithms'][:5]
print("Best algorithms:", [alg for alg, rating in top_algorithms])
```

## How It Works

### 1. Warmup Phase
- Tests all 22 algorithms on multiple diverse problems
- Each algorithm gets equal evaluation budget per problem
- Builds initial Elo ratings based on relative performance

### 2. Adaptive Phase  
- Focuses evaluation budget on the most promising algorithms
- Continues to update Elo ratings as more data comes in
- Balances exploration of new problems with exploitation of known strengths

### 3. Recommendation Engine
- Provides algorithm suggestions based on:
  - Current Elo ratings
  - Problem dimensionality 
  - Problem characteristics (smooth, multimodal, noisy)

## API Reference

### `adaptive_optimize()`

The main function that orchestrates the learning process.

```python
adaptive_optimize(
    objective_generator,    # Generator yielding objective functions
    trials_budget,         # Total evaluation budget
    n_dim,                # Problem dimension
    n_warmup_problems=5,   # Problems for initial testing
    trials_per_warmup=50,  # Evaluations per algorithm per problem
    elo_ratings_file=None, # File to persist Elo ratings
    verbose=True          # Print progress information
)
```

**Returns**: Dictionary with results including:
- `elo_system`: The EloRatingSystem with current ratings
- `top_algorithms`: List of (algorithm, rating) tuples
- `recommendations`: Algorithm suggestions by problem type
- `total_problems_solved`: Number of problems tested

### `EloRatingSystem`

Manages skill ratings for optimization algorithms.

```python
elo_system = EloRatingSystem()

# Get current rating
rating = elo_system.get_rating('NelderMead')

# Update ratings after a match (score_a: 1.0=win, 0.5=tie, 0.0=loss)
elo_system.update_ratings('NelderMead', 'RandomSearch', 1.0)

# Get top performers
top_10 = elo_system.get_top_algorithms(10)

# Persist ratings
elo_system.save_ratings('my_ratings.json')
elo_system.load_ratings('my_ratings.json')
```

### `suggest_algorithm_from_elo()`

Get algorithm suggestions based on learned performance.

```python
suggested = suggest_algorithm_from_elo(
    elo_system,           # Trained EloRatingSystem
    n_dim=5,             # Problem dimension
    problem_type='smooth' # 'smooth', 'multimodal', 'noisy', 'general'
)
```

## Objective Generators

The system requires generators that yield diverse optimization problems. Several are provided:

### Built-in Generators

```python
from humpday import sphere_variants_generator, rosenbrock_variants_generator

# Sphere function variants (good for testing global optimization)
sphere_gen = sphere_variants_generator(n_dim=3)

# Rosenbrock variants (good for testing local optimization)
rosenbrock_gen = rosenbrock_variants_generator(n_dim=3)
```

### Custom Generators

Create your own generators to test algorithms on problems relevant to your domain:

```python
def my_problem_generator(n_dim):
    """Generator for domain-specific optimization problems."""
    
    def problem_1(x):
        # Your first type of problem
        return np.sum((x - 0.3)**2)
    
    def problem_2(x): 
        # Your second type of problem
        return np.sum(x**4 - 16*x**2 + 5*x)
    
    problems = [problem_1, problem_2]
    
    while True:
        # Yield problems with random variations
        base = np.random.choice(problems)
        shift = np.random.uniform(-0.2, 0.2, n_dim)
        
        def shifted_problem(x, b=base, s=shift):
            x = np.asarray(x) + s
            return b(x)
            
        yield shifted_problem
```

## Example Use Cases

### Research and Development

```python
# Test algorithms on problems similar to yours
my_generator = create_my_domain_generator()

results = adaptive_optimize(
    objective_generator=my_generator,
    trials_budget=5000,
    n_dim=10,
    elo_ratings_file='research_elos.json'
)

# Use best-performing algorithms for your actual problems
top_alg = results['top_algorithms'][0][0]
best_val, best_x = pure_optimize(my_real_objective, top_alg, 200, 10)
```

### Automated Hyperparameter Tuning

```python
def hyperparameter_generator(param_dims):
    """Generate diverse hyperparameter optimization problems."""
    while True:
        # Generate different ML model training objectives
        yield create_random_ml_objective()

# Learn which optimizers work best for hyperparameter tuning
results = adaptive_optimize(
    objective_generator=hyperparameter_generator(param_dims),
    trials_budget=10000,
    n_dim=param_dims
)
```

### Continuous Learning

```python
# Load existing ratings
elo_system = EloRatingSystem()
elo_system.load_ratings('production_elos.json')

# Continue learning with new problems
results = adaptive_optimize(
    objective_generator=new_problem_generator(),
    trials_budget=1000,
    n_dim=5,
    elo_ratings_file='production_elos.json'  # Automatically saves updates
)
```

## Performance Insights

The Elo system reveals interesting patterns:

- **PRIMA algorithms** excel on smooth, low-dimensional problems
- **CMA-ES** dominates on multimodal, medium-dimensional problems  
- **Differential Evolution** performs consistently across problem types
- **Random Search** provides a useful baseline but rarely tops rankings
- **Particle Swarm** shows good performance on certain problem structures

## Integration with Existing Code

The adaptive system integrates seamlessly with existing Humpday usage:

```python
# Traditional usage
best_val, best_x = pure_optimize(objective, 'NelderMead', 100, 2)

# Adaptive usage - automatically selects best algorithm
elo_system = load_my_trained_elo_system()
suggested_alg = suggest_algorithm_from_elo(elo_system, n_dim=2, problem_type='smooth')
best_val, best_x = pure_optimize(objective, suggested_alg, 100, 2)
```

## Advanced Features

### Custom Scoring

Customize how algorithm performance is evaluated:

```python
def custom_tournament(algorithms, problems):
    """Custom tournament with domain-specific scoring."""
    # Your custom comparison logic
    pass
```

### Problem Classification

Train separate Elo systems for different problem classes:

```python
smooth_elo = train_on_smooth_problems()
multimodal_elo = train_on_multimodal_problems()

# Use appropriate system based on problem detection
if is_smooth(my_problem):
    alg = suggest_algorithm_from_elo(smooth_elo, n_dim, 'smooth')
else:
    alg = suggest_algorithm_from_elo(multimodal_elo, n_dim, 'multimodal')
```

This adaptive optimization system transforms algorithm selection from guesswork into data-driven decision making, continuously improving recommendations based on actual performance data.