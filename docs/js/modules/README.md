# humpday

23 derivative-free optimizers in zero-dependency JavaScript. This is the
browser/Node twin of the [Python humpday
package](https://github.com/microprediction/humpday); parity tests in the
repository hold the two implementations in agreement, so the algorithm you
run here is the algorithm the Python package ships.

Every optimizer minimises a black-box function on the unit cube `[0,1]^n`
under a hard evaluation budget. No gradients, no dependencies, no build
step.

## Install

```bash
npm install humpday
```

## Use

```js
const { Alloy, NelderMead, DifferentialEvolution } = require('humpday');

const objective = (x) => (x[0] - 0.3) ** 2 + Math.abs(x[1] - 0.6);

const opt = new Alloy(objective, 200, 2);   // objective, nTrials, nDim
const { bestValue, bestX } = opt.optimize();
```

Or create by name:

```js
const humpday = require('humpday');
const opt = new humpday.algorithms['CMAEvolutionStrategy'](objective, 200, 2);
```

## The roster

PRIMA trust-region (UOBYQA, NEWUOA, BOBYQA), classic numerical
(NelderMead, Powell, LBFGSB), evolutionary and swarm (DifferentialEvolution,
ParticleSwarm, GeneticAlgorithm, CMAEvolutionStrategy, EvolutionStrategy),
metaheuristics (SimulatedAnnealing, FireflyAlgorithm, AntColonyOpt,
HarmonySearch), model-based (BayesianOpt), local and pattern search
(Rechenberg, HillClimbing, CoordinateDescent, PatternSearch), baselines
(RandomSearch, GridSearch), and Alloy, a machine-designed blend of five
classical methods validated on held-out problems (see the
[papers](https://humpday.microprediction.org/papers.html)).

## Docs and live demos

Every optimizer can be raced in the browser on real physics and engineering
problems at [humpday.microprediction.org](https://humpday.microprediction.org).

MIT licensed.
