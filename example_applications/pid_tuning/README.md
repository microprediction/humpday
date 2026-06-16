# PID Controller Tuning

Tune the three gains `(Kp, Ki, Kd)` of a textbook PID controller so that an
underdamped second-order plant tracks a unit-step reference with minimal
integral-of-squared-error (ISE).

The plant is the classic mass–spring–damper system

    y'' + 2·ζ·ωn·y' + ωn²·y = ωn²·u

with `ζ = 0.3`, `ωn = 1.0` (lightly damped — it overshoots and rings if
controlled badly). The loop is closed with `u = Kp·e + Ki·∫e + Kd·de/dt`,
`e = r − y`, `r = 1`, integrated with explicit Euler (`dt = 0.01`, horizon
`T = 30`). A competent tuning reaches ISE ≈ **1–3**.

## What this stresses

- **Ill-conditioned valley.** `Kp` and `Kd` trade off (more derivative
  damping permits more proportional gain) and `Ki` couples in to erase
  steady-state error, so the good region is a curved, skewed ravine, not an
  axis-aligned bowl.

- **Instability cliff.** Gains that are too large or imbalanced make the
  closed loop diverge. Rather than overflow, the simulation stops and returns
  a flat `PENALTY = 1e3`. That high plateau borders the stable basin, so an
  optimiser must descend into the ravine without stepping off the edge.

## Variables

| Gain | Symbol | Range |
|---|---|---|
| Proportional | `Kp` | 0 – 30 |
| Integral     | `Ki` | 0 – 15 |
| Derivative   | `Kd` | 0 – 15 |

## Running

```bash
python -m example_applications.pid_tuning.run
```

Output is a small comparison table of optimiser → best ISE → gains. Good
optimisers find stable tunings with ISE ≈ 1–3; weaker runs get stranded on
the unstable penalty plateau (1e3).
