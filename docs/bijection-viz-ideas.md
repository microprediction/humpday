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
(both are 2-balls) but **not** diffeomorphic as manifolds-with-corners. φθ
resolves this by sending the boundary "to infinity": as you push toward a cube
corner the image races to the simplex edge and the grid stretches without bound.
So markings near the square's border (pitch corners, touchlines) smear along the
simplex boundary rather than land on tidy corners — a feature to lean into.

### Precise statement and reference
Regard `X = [0,1]²` and the standard 2-simplex `Δ² = { t ∈ ℝ³₊ : Σ tᵢ = 1 }` as
smooth **manifolds with corners**. *Claim:* there is no diffeomorphism (of
manifolds with corners) `X → Δ²`.

The citable invariant is **depth**. Every point `x` of an `n`-dimensional
manifold with corners `M` has a well-defined depth `depthₘ(x) ∈ {0,…,n}`: in any
chart to the model `ℝⁿ_k = [0,∞)ᵏ × ℝⁿ⁻ᵏ` sending `x` to the corner of the model,
`k = depthₘ(x)`. The depth strata and `k`-corners `C_k(M)` are **intrinsic**
(atlas-independent), hence preserved by diffeomorphisms.

- **D. Joyce, "On manifolds with corners," §2** — definitions of depth, boundary
  `∂M`, corners `C_k(M)`, and their invariance. arXiv:0910.3518
  (https://arxiv.org/abs/0910.3518); *Advances in Geometric Analysis*, Adv. Lect.
  Math. **21** (2012) 225–258.
- **J. Margalef-Roig & E. Outerelo Domínguez, *Differential Topology*,**
  North-Holland Math. Studies **173** (1992) — textbook proof that the *index*
  (= depth) of a point is a differentiable invariant.

*Proof.* A diffeomorphism `F : X → Δ²` preserves depth, so restricts to a
bijection of depth-2 strata `S₂(X) → S₂(Δ²)`. But `|S₂([0,1]²)| = 4` (corners)
and `|S₂(Δ²)| = 3` (vertices); `4 ≠ 3`. ∎

(Honest caveat: no paper states "square ≇ triangle" as a headline result — it is
the immediate corner-count corollary of the cited depth-invariance.)
