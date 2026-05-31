# Embedding the 3D optimization visualizer

The same in-browser 3D demo that runs on every page at
[humpday.microprediction.org/algorithms/](https://humpday.microprediction.org/algorithms/)
can be dropped into any HTML page. It's all static — no build step, no
package install, no server.

## Drop-in HTML

```html
<!-- Three.js (3D rendering) -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

<!-- HumpDay algorithms (pure-JS ports, no dependencies) -->
<script src="https://microprediction.github.io/humpday/js/modules/base-optimizer.js"></script>
<script src="https://microprediction.github.io/humpday/js/modules/prima-algorithms.js"></script>
<script src="https://microprediction.github.io/humpday/js/modules/scipy-algorithms.js"></script>
<script src="https://microprediction.github.io/humpday/js/modules/evolutionary-algorithms.js"></script>
<script src="https://microprediction.github.io/humpday/js/modules/search-algorithms.js"></script>
<script src="https://microprediction.github.io/humpday/js/modules/index.js"></script>

<!-- The visualizer itself -->
<script src="https://microprediction.github.io/humpday/js/algorithm-visualizer.js"></script>

<!-- Where the visualization renders -->
<div id="optimizationDemo" style="width: 100%; height: 500px; border: 1px solid #ddd;"></div>

<!-- Initialize once the page is ready -->
<script>
document.addEventListener('DOMContentLoaded', function () {
    new AlgorithmVisualizer('optimizationDemo');
});
</script>
```

## What you get

- **22 derivative-free optimizers** — the same ones HumpDay ships in
  Python (PRIMA UOBYQA/NEWUOA/BOBYQA, NelderMead, Powell, LBFGSB,
  DifferentialEvolution, ParticleSwarm, SimulatedAnnealing,
  GeneticAlgorithm, RandomSearch, BayesianOpt, CMAEvolutionStrategy,
  FireflyAlgorithm, AntColonyOpt, EvolutionStrategy,
  HillClimbing, HarmonySearch, Rechenberg, CoordinateDescent,
  PatternSearch).
- **12 test surfaces** — Sphere, Rosenbrock, Ackley, Rastrigin,
  Griewank, Schwefel, Beale, Booth, Himmelblau, Matyas, Levy, Easom.
- **3D view** with rotate / zoom / pan via OrbitControls.
- **Speed control** and algorithm-vs-algorithm comparison.

## Constructor options

```js
new AlgorithmVisualizer('optimizationDemo', {
    onReady: () => { console.log('visualizer ready'); },
});
```

The visualizer reads its container's size, so it stretches to whatever
the surrounding `<div>` is — set the width and height on the div.

## Self-hosting

If you'd rather host the JS yourself (instead of pointing to the
GitHub Pages CDN), every file under [docs/js/](./js/) in this repo
is MIT-licensed. Clone, copy what you need.

## Issues

The visualizer's source is in
[`docs/js/algorithm-visualizer.js`](./js/algorithm-visualizer.js).
File bugs at <https://github.com/microprediction/humpday/issues>.
