# Wind Farm — Wake-Coupled Layout

Place 8 turbines in a field to maximise expected power over a 12-sector wind rose.
HumpDay's `[0,1]^16` cube sets each turbine's position; the objective sums power
across wind directions using the **Jensen wake model** (turbines in another's
wake see slower wind, and power scales with wind speed cubed), minus a minimum-
spacing penalty.

## What this stresses

- **A non-separable objective.** Every turbine's output depends on where the
  others are (through their wakes), so you can't optimise positions
  independently.
- **Directional structure.** The wind rose has prevailing directions; good
  layouts stagger turbines so few sit directly downwind of another.
- **Soft + hard terms.** A spacing penalty and a spread bonus shape a layout that
  is both legal and efficient.

## Running

```bash
python -m example_applications.wind_farm.run
```

Expect the best layouts to reach ~93–95% of the wake-free power. Mirrors the
browser demo [`docs/applications/wind-farm.html`](../../docs/applications/wind-farm.html).
