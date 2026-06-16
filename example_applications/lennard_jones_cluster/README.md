# Lennard-Jones Cluster (N = 5)

A canonical *extreme-multimodality* global-optimisation benchmark from
computational chemistry. Place 5 atoms in 3-D space and minimise the
total Lennard-Jones pair potential

    E = sum_{i<j} 4 * (r_ij**-12 - r_ij**-6),    r_ij = |x_i - x_j|.

The 15 coordinates (3 per atom) are mapped from `[0,1]^15` into a cubic
box `[0.0, 3.0]^3`, so `N_DIM = 15`. Pairwise distances are clamped to a
floor of `0.1` so overlapping atoms give a large but finite positive
energy instead of an overflow.

The known global minimum for `N = 5` is a **triangular bipyramid** with
energy ≈ **-9.103852**. Reaching it in a few hundred evaluations is hard;
a good optimiser should land in roughly the `-7` to `-9` range.

## What this stresses

- **Exponential multimodality.** The count of distinct local minima of
  the LJ potential grows roughly exponentially with the number of atoms.
  Even at `N = 5` the landscape is a dense forest of funnels, so
  optimisers that converge greedily get trapped far from the ground
  state.

- **Gauge freedom.** Energy is invariant under any global translation or
  rotation of the cluster, so every optimum is a continuum of equivalent
  configurations. This flat manifold of equally-good solutions confuses
  methods that assume isolated optima.

- **Stiff repulsive wall.** The `r**-12` term makes the potential extremely
  steep where atoms approach each other, producing near-vertical cliffs
  next to deep wells — a brutal mix of scales for any search heuristic.

## Running

```bash
python -m example_applications.lennard_jones_cluster.run
```

Output is a small comparison table of optimiser → best energy. The
population-based and trust-region methods tend to reach the deepest
(most negative) energies within the 800-trial budget.
