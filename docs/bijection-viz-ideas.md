# Cube ⇆ simplex visualization — idea backlog

The interactive page [`simplex-cube-bijection.html`](simplex-cube-bijection.html)
renders the map **φθ** between the unit square and the probability simplex. A few
directions queued for follow-up (social / teaching hooks):

## Transport an object across the map ("swivel" anything onto the simplex)
The page currently morphs a *coordinate net*. The same lift can carry **any**
picture on the cube onto the simplex:

- **An objective field.** Drape a level-set heatmap (or contour lines) of a cube
  objective and watch it deform into the simplex — directly showing how the
  bijection reshapes a landscape (the preconditioner story, made visual).
- **An optimizer's trajectory.** Replay a real `pure_optimize` search path on the
  cube and show the corresponding path on the simplex (and how θ changes it).
- **A "swivel" / vector field.** Push a flow or rotating field through φθ to show
  the local distortion (Jacobian) — where the map stretches vs compresses.

## FIFA theme — football markings on the simplex ✅ built
Built: [`soccer-field-simplex.html`](soccer-field-simplex.html). A regulation
pitch (halfway line, centre circle, penalty boxes, arcs, spots, mowing stripes)
on the unit square, pushed through φθ onto the simplex — straight lines bend, the
centre circle becomes an oval, the corners smear toward the boundary. Same 3D
engine (orbit, morph, presets θ₀ / θ★ / degenerate `STD_L=500`). Social assets:
`assets/soccer-field-simplex.gif`, `assets/video/soccer-field-simplex.mp4`,
`assets/soccer-field-simplex.png`.

## Already in the page
- Preset story: `θ₀` default → learned `θ★` → degenerate `STD_L = 500` (the whole
  square collapses onto the centroid — the dependency/scale bug we fixed).
- Objective-field colour mode with the optimum's pre-image readout.

## A note for any of these — it's only an *interior* diffeomorphism
φθ is a diffeomorphism between the **open** cube and the **open** simplex (both ≅
ℝⁿ via probit then log-ratio). The *closed* square and triangle are homeomorphic
(both are 2-balls) but **not** diffeomorphic as manifolds-with-corners — their
corner combinatorics differ (a square has 4 vertices/2 edges each; a triangle has
3). φθ resolves this by sending the boundary "to infinity": as you push toward a
cube corner the image races to the simplex edge and the grid stretches without
bound. So markings near the square's border (pitch corners, touchlines) will
smear along the simplex boundary rather than land on tidy corners — a feature to
lean into, not hide.
