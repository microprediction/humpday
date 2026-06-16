# HumpDay Example Applications

Real-world(ish) case studies that exercise HumpDay's pure-Python optimisers
on the kind of objectives derivative-free optimization is actually for —
not the classical analytic benchmarks (Rastrigin, Griewank, Salomon) but
problems whose pathologies (constraints, noise, non-stationarity,
expensive evaluation) are why anyone reaches for black-box methods in
the first place.

Each subfolder is self-contained:

| Folder | Domain | What it stresses |
|---|---|---|
| [`welded_beam/`](welded_beam/)             | Structural engineering            | Constraint handling via penalty functions; navigation of narrow feasible regions on disparate dimension scales. |
| [`cart_pole_policy/`](cart_pole_policy/)   | Reinforcement learning            | Direct policy search; tolerance to stochastic episode noise; medium-dimensional parameter spaces. |
| [`algo_trading/`](algo_trading/)           | Quantitative finance              | Walk-forward optimisation; in-sample vs out-of-sample generalisation; the danger of overfitting sharp peaks. |
| [`airfoil_shape/`](airfoil_shape/)         | Surrogate-based aerodynamics      | Severely limited evaluation budgets; landscapes where Bayesian methods dominate evolutionary ones. |
| [`lens_design/`](lens_design/)             | Optical ray tracing               | Rugged, deceptive landscape; a sharp-focus "needle" random search can't find; interpolation/local methods win. |
| [`genetic_art/`](genetic_art/)             | Generative image fit              | High dimensionality; population methods shine while local methods collapse — the reverse of `lens_design`. |
| [`tetris_weights/`](tetris_weights/)       | Game-heuristic tuning             | Noisy objective; in-sample vs out-of-sample overfitting; a non-textbook optimum. |
| [`plinko_funnel/`](plinko_funnel/)         | Stochastic process control        | Pure evaluation noise; steering a random cascade onto an off-centre target. |
| [`walking_creature/`](walking_creature/)   | Evolved locomotion                | Emergent structure from a scalar reward (an alternating gait); a broad basin with interpolation traps. |
| [`ebola_response/`](ebola_response/)       | Epidemic control                  | Multi-objective harm (deaths vs cost) folded into one scalar; the optimum is a *schedule*, not a setting. |
| [`espresso_dialin/`](espresso_dialin/)     | Sample-efficient tuning           | Tiny noisy budget; interpolation / Bayesian methods win where population methods are still warming up. |
| [`fm_sound_match/`](fm_sound_match/)       | Audio / spectral inverse          | Recover a synth patch from its spectrum; smooth here, but a famously octave-trapped problem if unbounded. |
| [`boids_flocking/`](boids_flocking/)       | Emergent navigation               | A swarm threading a chicane; a broad, forgiving basin where even Random Search does well. |
| [`antenna_array/`](antenna_array/)         | Antenna design                    | Optimiser beats intuition: an irregular element spacing beats the even array; a wiggly multimodal landscape. |
| [`circle_packing/`](circle_packing/)       | Packing geometry                  | Non-smooth "maximise the minimum"; sharp ridges where the binding gap switches; a known optimum. |
| [`brachistochrone/`](brachistochrone/)     | Calculus of variations            | The fastest-slide curve (a cycloid, not the straight line); a counter-intuitive smooth optimum. |
| [`tuned_mass_damper/`](tuned_mass_damper/) | Mixed-integer seismic             | Three continuous knobs plus one integer floor; dynamics (Newmark-β) in the objective; resonance tuning. |
| [`tennis_doubles/`](tennis_doubles/)       | Game strategy                     | Tune doubles tactics vs a textbook team; noisy, in/out-of-sample overfitting; safe play wins. |
| [`chess_piece_values/`](chess_piece_values/) | Game strategy (expensive)       | A perft-verified depth-2 engine; objective-design cautionary tale — it exploits the opponent, not the textbook values. |
| [`robot_arm/`](robot_arm/)                 | Inverse kinematics                | Hard collision constraints; disjoint elbow-up/down/wrap solution branches through narrow corridors. |
| [`wind_farm/`](wind_farm/)                 | Energy layout                     | Wake-coupled, non-separable objective; spread turbines out of each other's wakes under a spacing limit. |
| [`rocket_landing/`](rocket_landing/)       | Optimal control                   | Multimodal throttle schedule (gradual descent vs late suicide burn); a tight fuel budget and a crash cliff. |
| [`battery_dispatch/`](battery_dispatch/)   | Energy arbitrage                  | Price arbitrage under state-of-charge limits and round-trip efficiency losses; over-trading destroys value. |
| [`reactor_profile/`](reactor_profile/)     | Reaction engineering              | Optimal control of an A→B→C reactor; the best temperature is a *profile*, not a constant. |
| [`bridge_truss/`](bridge_truss/)           | Structural optimisation           | FEM-in-the-loop member sizing; lightest truss on the yield/buckling boundary; statically indeterminate. |
| [`free_kick/`](free_kick/)                 | Sports ballistics                 | 3-D ball flight (Magnus curve) past a wall and a diving keeper; multimodal goal/save/block outcomes. |
| [`bowling/`](bowling/)                     | Chain-reaction physics            | Faithful 105-pin collision sim; rough, sensitive landscape where small entry changes swing the count. |
| [`trebuchet/`](trebuchet/)                 | Ballistics (reduced-order)        | Hit a target 60 m away; interior efficiency optimum in arm/sling ratios. *Simplified — demo uses Matter.js.* |
| [`curling/`](curling/)                     | Slide-to-target (reduced-order)   | Stop the stone on the button; too little weight stops short, too much sails through. *Simplified.* |
| [`mini_golf/`](mini_golf/)                 | Putt-to-hole (reduced-order)      | Read the slope and sink the putt. *Simplified — demo uses Matter.js.* |
| [`pool/`](pool/)                           | Cut-shot aim (reduced-order)      | The ghost-ball cut angle is a narrow, precise optimum. *Simplified single-cut model.* |
| [`slingshot/`](slingshot/)                 | Ballistics (reduced-order)        | Rake one block stack or loft onto the other — two basins. *Simplified — demo uses Matter.js.* |
| [`cocktail_blend/`](cocktail_blend/)       | Composition / inverse problem     | Proportions on the unit simplex (sum to 1) via the cube→simplex bijection; an over-determined flavour target with an interior optimum. |
| [`portfolio_frontier/`](portfolio_frontier/) | Composition / finance (non-convex) | Long-only weights on the unit simplex via the same bijection; a cardinality cost makes it non-convex and bimodal — a corner trap vs a diversified basin that ranks optimisers. |

Of these, twenty-eight (`boids_flocking` through `slingshot`) mirror, in pure
Python, the interactive browser demos at
[`docs/applications/`](../docs/applications/). Several are deliberately a matched
pair — `lens_design` (low-D, interpolation/local methods win) versus
`genetic_art` (high-D, those same methods lose) — so the **No-Free-Lunch**
lesson is reproducible at the command line, not just asserted. `cocktail_blend`
and `portfolio_frontier` have no browser counterpart: they are the
simplex-geometry case studies for the cube→simplex bijection — the first smooth
(it demonstrates the lift), the second non-convex (it ranks optimisers).

## The HumpDay convention

Every objective in HumpDay maps the unit hypercube `[0, 1]^n` to a
real-valued cost to minimise. So each example's `problem.py` does two
things:

1. **Scale** the `[0, 1]^n` hypercube point to whatever physical units
   the application naturally lives in.
2. **Evaluate** the application-specific cost (with constraint penalties,
   episode rollouts, backtests, surrogate calls, etc).

The `run.py` in each folder calls a handful of representative
optimisers from `humpday.optimizers.alloptimizers.PURE_OPTIMIZERS` —
typically one local (NelderMead), one population-based (DifferentialEvolution
or CMA), one Bayesian (BayesianOpt), and one trust-region (PRIMA_BOBYQA) —
and prints a small comparison table.

## Running an example

From the repo root:

```bash
python -m example_applications.welded_beam.run
python -m example_applications.cart_pole_policy.run
python -m example_applications.algo_trading.run
python -m example_applications.airfoil_shape.run
python -m example_applications.lens_design.run
python -m example_applications.genetic_art.run
python -m example_applications.tetris_weights.run
python -m example_applications.plinko_funnel.run
python -m example_applications.walking_creature.run
python -m example_applications.ebola_response.run
python -m example_applications.espresso_dialin.run
python -m example_applications.fm_sound_match.run
python -m example_applications.boids_flocking.run
python -m example_applications.antenna_array.run
python -m example_applications.circle_packing.run
python -m example_applications.brachistochrone.run
python -m example_applications.tuned_mass_damper.run
python -m example_applications.tennis_doubles.run
python -m example_applications.chess_piece_values.run
python -m example_applications.robot_arm.run
python -m example_applications.wind_farm.run
python -m example_applications.rocket_landing.run
python -m example_applications.battery_dispatch.run
python -m example_applications.reactor_profile.run
python -m example_applications.bridge_truss.run
python -m example_applications.free_kick.run
python -m example_applications.bowling.run
python -m example_applications.trebuchet.run
python -m example_applications.curling.run
python -m example_applications.mini_golf.run
python -m example_applications.pool.run
python -m example_applications.slingshot.run
python -m example_applications.cocktail_blend.run
python -m example_applications.portfolio_frontier.run
```

No external dependencies beyond what HumpDay itself uses — these are
intentionally pure-Python so they work in browser/Pyodide too.

## Why these four?

Together they span the typological cross-section that the HumpDay
recommender system needs to be calibrated against:

- **Constraints with hard physical limits** (welded beam): penalty cliffs
  the optimiser must descend without falling off.
- **Stochastic objective** (cart pole): a single evaluation gives a noisy
  estimate; an algorithm that over-trusts a single low value will be
  punished out-of-sample.
- **Non-stationary objective** (algo trading): the in-sample optimum is
  not the out-of-sample optimum; this distinguishes algorithms that hunt
  sharp peaks from those that prefer broad flats.
- **Expensive objective** (airfoil): every evaluation is precious; the
  algorithm must extract maximum information from few samples.

These four are the ones an industrial user would actually face. The
Elo rankings derived from these tasks will be more useful than those
derived from purely analytic functions.
