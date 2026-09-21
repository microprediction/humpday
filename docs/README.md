# HumpDay on the web

This directory is the site published at [humpday.microprediction.org](https://humpday.microprediction.org):
the optimizer contest, the per-algorithm pages, and the application demonstrations.

## How it works

The optimizers run as native JavaScript, in `js/modules/`. There is no Python in the browser: the
page loads `prng.js`, `base-optimizer.js` and the algorithm modules as ordinary script tags, and
every algorithm is a port of the Python one rather than a call into it. Thirteen of the
twenty-three are bit-exact twins, replaying `parity/transition_vectors.json` point for point; the
rest agree on behaviour but not on every last bit, and #78 tracks the difference.

An earlier version of this demo ran CPython in the browser through Pyodide and called SciPy. That
is gone, along with the download it required.

## Pages

- `contest.html` races the optimizers against each other on a chosen objective.
- `algorithms.html` and `algorithms/` explain one algorithm each, with a live visualization.
- `applications/` works a real problem per page, with the objective written out.
- `recommendations.html` reads the recorded tournament: which optimizer for which dimension and
  budget.

## Local development

```bash
python -m http.server 8000     # from this directory
# or
npx serve .
```

Then open http://localhost:8000. A server rather than `file://`, because the pages fetch their
modules and their data.

## Keeping it honest

The parity tests in `tests/` compare the JavaScript against the Python, replaying the recorded
transition vectors through both. Run them with `pytest tests -q` from the repository root.
