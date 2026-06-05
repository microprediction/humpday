# Robot Arm — Constrained Inverse Kinematics

Put the tip of a 6-link planar arm on a target while keeping every link clear of
three circular obstacles. HumpDay's `[0,1]^6` cube sets the six joint angles
(each ±90°); the objective scores reach (100 at touch, falling with distance)
minus a heavy penalty for any link intersecting an obstacle.

## What this stresses

- **Hard constraints.** Collisions are penalised steeply, so the feasible region
  is carved by penalty cliffs the optimiser must descend without falling off.
- **Disjoint solution branches.** Elbow-up, elbow-down and wrap-around poses all
  reach the target through different, narrow collision-free corridors — a
  multimodal landscape.
- **Naive poses crash.** A random arm usually punches a link through an obstacle;
  finding the corridor is the whole game.

## Running

```bash
python -m example_applications.robot_arm.run
```

Expect Differential Evolution to land the tip ~1 px from target with no
collision, while methods that get stuck against the cliffs sit short. Mirrors the
browser demo [`docs/applications/robot-arm.html`](../../docs/applications/robot-arm.html).
