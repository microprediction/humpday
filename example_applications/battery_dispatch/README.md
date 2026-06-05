# Battery Dispatch — Price Arbitrage Under Constraints

Arbitrage a day of electricity prices with a grid battery. HumpDay's `[0,1]^24`
cube is the hourly charge/discharge power; the objective clips each hour to what
the state-of-charge and power limits allow, applies round-trip efficiency losses,
and returns the negative **daily revenue** against a fixed price curve.

## What this stresses

- **Linear dynamics with operational limits.** State-of-charge bounds and a power
  cap make many naive schedules infeasible (and silently clipped), so the
  optimiser must respect the pack.
- **An efficiency tax on every cycle.** Round-trip losses mean over-trading
  destroys value — the optimum cycles just enough.
- **Exploiting the price shape.** Buy low overnight, ride the midday dip, sell
  into the sharp evening peak.

## Running

```bash
python -m example_applications.battery_dispatch.run
```

The naive one-cycle preset earns ~$13.7k/day; good schedules do better. Mirrors
the browser demo [`docs/applications/battery-dispatch.html`](../../docs/applications/battery-dispatch.html).
