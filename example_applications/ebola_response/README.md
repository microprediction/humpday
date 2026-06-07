# Ebola Response — Multi-Objective Control Timing

Time a public-health intervention to minimise harm. A pure-Python **SEIR**
epidemic model runs for 180 days split into 8 control windows; HumpDay's
`[0,1]^8` cube sets the control intensity in each window. Stronger control
suppresses transmission but costs economic effort, so harm = deaths + cost. The
objective returns the negative **percentage of harm avoided** versus doing nothing.

## What this stresses

- **Multi-objective in disguise.** Two opposing costs (lives vs effort) folded
  into one scalar; the optimum is a *schedule*, not a single setting.
- **Control timing.** Spend too little and the epidemic burns through the
  population; spend too much, too early, and you pay for control the outbreak
  didn't need yet. Good policies stay low, spike during the growth phase, then
  relax — the epidemiologist's playbook, learned from a scalar reward.
- **Smooth, moderate dimension (8-D).** Tractable for most methods; the table
  shows who finds the ~82% sweet spot and who settles lower.

## Running

```bash
python -m example_applications.ebola_response.run
```

The control-profile column shows each window as `.`/`+`/`#` (little/moderate/hard)
— watch the good policies concentrate effort in the middle. Mirrors the browser
demo [`docs/applications/ebola.html`](../../docs/applications/ebola.html).
