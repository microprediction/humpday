# Direct Policy Search — CartPole

Train a 5-parameter linear controller to balance an inverted pendulum
on a moving cart, by **direct search over the policy weights**. No
gradients, no value functions, no rollouts-with-backprop — just a
black-box optimiser hunting a 5-D parameter vector whose only output
is the average return across a handful of stochastic episodes.

The classic CartPole-v1 environment runs for up to 500 steps; we
replicate its dynamics in ~30 lines of pure Python (no `gym` dependency)
so this example works in browser/Pyodide too. The reward per episode
is the number of steps survived; the optimisation objective is the
**negative mean return** across `N_EPISODES=8` rollouts with different
random seeds.

## What this stresses

- **Stochastic objective.** Each evaluation runs 8 episodes with
  random initial conditions; the same parameter vector produces a
  different cost on every call. Optimisers that over-trust a single
  low value get fooled.

- **Plateau + cliff topology.** Below some "stable" threshold of
  policy quality the cart always falls quickly (cost ≈ −20 to −50
  consistently). Above the threshold the cart balances for the full
  500 steps (cost ≈ −500). The transition is sharp. The optimiser
  must escape the plateau without getting stuck in tiny local-cost
  improvements.

- **Medium-low dimension (5-D).** Tractable for any reasonable
  algorithm, but exposes whether population-based methods (which need
  to fill a 5-D ball) can compete with local + Bayesian methods on
  small but noisy problems.

## The policy

A linear controller:

    action = step  if  w · state + b > 0  else  pull_back

The 5 parameters are `(w1, w2, w3, w4, b)` — three weights on cart
position/velocity/pole-angle/pole-rate, plus a bias. HumpDay's `[0,1]^5`
unit cube is stretched to `[-5, 5]^5` via a centred affine map.

## Why direct search beats gradients here

The reward function is `Σ 1_{cart_alive}` — a sum of step indicators.
It is not differentiable in the policy parameters; any "gradient" must
be estimated by perturbation. In practice that means policy gradients
have high variance, and methods like REINFORCE need millions of samples
to find what a well-chosen black-box optimiser can find in a few
hundred.

## Running

```bash
python -m example_applications.cart_pole_policy.run
```

Expect to see CMA-ES, DifferentialEvolution, and well-tuned local
methods consistently find policies with mean return >= 400 (out of
500). The output table includes the *best* and the *median* return
of the final policy across an independent test set of 20 episodes —
the difference between those numbers is the noise-overfitting tax.
