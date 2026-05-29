# HumpDay applications — build guidelines

Patterns and pitfalls collected from the bowling / cart-pole / slingshot /
mini-golf / pool builds. Read this before starting a new demo. The intent
is to keep the demos visually consistent, to stop user feedback from
repeating itself across demos ("more power!", "format is off!", "balls
don't come to rest!"), and to keep me honest about what the optimisers
actually do.

The slingshot demo at `docs/applications/slingshot.html` is the
canonical template. When in doubt, copy from there.

---

## 0. Workflow at a glance

For a new demo `xyz`:

1. `git checkout main && git pull && git checkout -b add-xyz-demo`
2. Copy `docs/applications/slingshot.html` → `docs/applications/xyz.html`,
   adapt the title/intro/physics/parameters.
3. **Add a card to `docs/applications/index.html`** in the "Live demos"
   grid (section 13 below). A demo that isn't on the index is invisible
   to anyone arriving at the applications page.
4. Local preview (section 10), iterate until it looks right.
5. Smoke-test the simulator with N random params (section 11) before
   writing copy that asserts anything about the landscape.
6. Walk the section 13 checklist.
7. Commit, push, open PR.

---

## 1. Page shell — copy from slingshot.html

Every demo uses the same HTML structure. Don't redesign it; the user
prefers consistency across demos.

```
<nav class="site-nav">  ... site links
<div class="container">
  <h1>EMOJI Name Optimizer</h1>
  <p class="intro">                ← one short paragraph, no overclaiming
  <div class="stage">
    <div class="left-stack">
      <canvas width="800" height="450"></canvas>     ← always 800×450
      <div class="panel hr-panel">  ← Human Raphson sliders
    </div>
    <div class="panel">             ← right column: algorithm + budget + run
  </div>
  <div class="leaderboard">         ← session leaderboard
  <h2>What's happening</h2>         ← short, honest description
  <p>...Physics powered by Matter.js... ← attribution if applicable
  <div class="skill-callout">       ← Save-the-Planet callout
</div>
```

CSS variables (`--bg`, `--panel`, `--accent`, etc.) are identical across
demos. Don't introduce per-demo theming except for the canvas background.

## 2. Required UI elements

Every demo MUST include:

- **Algorithm dropdown** with the same four optgroups in the same order:
  Trust-region/interpolation, Local/classical, Population-based, Other.
- **Budget select**: `10 / 20 / 50 / 100 / 200 / 500` (some demos also `1000`).
- **Run / Replay best / Reset leaderboard** buttons in that order.
- **Stats panel** on the right showing: Score (big), Detail (small),
  Evals tried, Best so far, plus a row per parameter.
- **Leaderboard** below the stage — algorithm | score | evals used | params.
- **Human Raphson sliders** — 3-column grid layout in `.hr-sliders`.
- **Save-the-Planet callout** with the SKILL.md copy-paste snippet.

## 3. Parameter envelope — slider and decode MUST match

> User caught this on the pool demo: slider was 5–70, decode was 8–64.

For every parameter, the slider's `min`/`max` and `decode(u)`'s
output range must be **identical**. The optimiser searches the slider's
exact envelope so the user knows what space is being explored.

```js
// GOOD — slider min=20, max=70, decode is power = 20 + 50 * u[1]
// BAD  — slider min=20, max=70, decode is power = 5 + 65 * u[1]
```

When you extend a slider, extend the decode in the same commit.

## 4. Physics conventions

### Matter.js (slingshot, mini-golf, pool, future demos)

- **friction**: kinetic friction coefficient. Higher = MORE drag.
- **frictionAir**: per-frame velocity damping. Higher = balls stop sooner.
- **restitution**: bounciness. 0 = inelastic, 1 = elastic.

### Custom velocity-multiplier sims (bowling)

In the bowling simulator, `friction` is applied as
`velocity *= friction` each step. So **higher = LESS drag**. This is the
OPPOSITE of Matter.js. Don't confuse the two.

### Sim length and rest detection

Low friction = balls roll a long time. Compute approximate stopping time
before setting `N_SIM_STEPS`. Rule of thumb: at frictionAir = 0.004 a
ball halves velocity every ~170 frames, so 900 frames (15 s) is the
minimum for a clean break to fully come to rest.

Use multi-check rest detection — single-check thresholds fire during
mid-roll lulls (e.g. the instant a ball reverses off a cushion):

```js
if (maxSpeed < REST_SPEED) { restConfirms++; if (restConfirms >= 3) break; }
else { restConfirms = 0; }
```

## 5. Animation

- `REPLAY_MS_PER_FRAME` ≈ 18–25 ms (slow final replay)
- `MONTAGE_MS_PER_FRAME` ≈ 9–12 ms (search-montage, roughly 2× replay)
- One `animationToken` counter; bump it before starting any new sim or
  the previous animation will keep firing into the next one.
- Capture the token at the start of an animation; don't increment inside
  `animateShot` (caused the slingshot "stops after 2 guesses" bug).

## 6. Update params on EVERY evaluation, not just `isNew`

> User has been bitten by this on slingshot, mini-golf, and cart-pole.

The Angle / Power / Spin / etc. stat rows should reflect the CURRENT
attempt's parameters, not just the most recent best. Otherwise the
numbers freeze on the first basin and look broken between improvements.

```js
// In the search montage, ALWAYS:
updateBestParams(entry.params, entry.score);
// And conditionally update best-score / leaderboard if isNew.
```

## 7. Stats panel formatting

- Big number (`.score`) gets just the integer. Long compound strings
  break the panel layout.
- Put breakdown ("3 in, scratch") in a separate Detail row.
- Algorithm name goes on a smaller second line via `<span class="algo-tag">`
  — concatenating "1 (1 in) — CMAEvolutionStrategy" wraps the
  "Best so far" label across two lines.
- `.stat-row span:first-child { white-space: nowrap; flex-shrink: 0; }`
  so labels never split.
- `.stat-row span:last-child { text-align: right; word-break: break-word; }`
  so long values wrap into the value column, not the label.

## 8. "What's happening" copy — don't overclaim

> User: "don't jump to conclusions" — caught me writing "most random
> shots sink zero" and "Local methods plateau on..." before running a
> single shot.

Describe the setup, the score function, and the parameter space.
DO NOT predict which algorithms will win unless you've measured it.
Invite the user to compare runs.

## 9. Local preview

```bash
python3 -m http.server 8765 --directory docs > /tmp/server.log 2>&1 &
open http://localhost:8765/applications/<demo>.html
```

The `<meta http-equiv="Content-Security-Policy">` blocks `file://`
loads of the JS modules, so a static server is required. Matter.js
CDN is whitelisted in the existing CSP — don't add other CDNs without
also widening the CSP.

## 10. Smoke-test before claiming behaviour

Extract the simulator from the HTML and run N=1000–2000 random shots
in Node (see the pool/bowling commits for the pattern). Report
**actual** best/mean scores before writing copy that asserts a
landscape property.

```js
// docs/applications/<demo>.html contains <script>... main game logic ...</script>
const html = fs.readFileSync('docs/applications/X.html', 'utf8');
const inline = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)]
  .map(m => m[1]).find(s => s.length > 1000);
// Stub DOM, extract decode + simulate, loop random params, report stats.
```

## 11. Adding the demo to the applications index — DO NOT SKIP

> A demo not linked from `docs/applications/index.html` is effectively
> invisible. This is the easiest step to forget and the user has
> reminded me twice now.

Open `docs/applications/index.html` and add an `<a class="card live">`
block to the **Live demos** grid (just before the closing `</div>` of
`<div class="grid">` that follows the `<h2>Live demos</h2>`). The card
template:

```html
<a class="card live" href="xyz.html" style="text-decoration:none;">
  <div class="emoji">🎯</div>                       <!-- pick a fitting emoji -->
  <span class="tag">Live</span>
  <h3>Demo Name</h3>
  <p class="desc">
    One-paragraph description. Mention the search space size (n=3, n=4),
    what's being optimised, and the score range. Keep it concise — this
    sits in a fixed-height card.
  </p>
  <div class="meta">
    <span>n=3 · score 0-100</span>                  <!-- left meta: search size + range -->
    <span class="play">Open →</span>
  </div>
</a>
```

Conventions:

- **Emoji**: a single emoji that maps to the demo (🎳 🎱 ⛳ 🦅 🛞 🚗).
- **Tag**: `<span class="tag">Live</span>` for finished demos. Use
  `<span class="tag soon">Coming</span>` on a `.card.coming-soon` for
  placeholders.
- **Desc**: <= 2 sentences. Do not predict algorithm behaviour here
  any more than you would in "What's happening" (see section 8).
- **Meta left cell**: `n=K · <unit>` — search-space dimensionality
  plus the score's natural unit/range.
- **Page browser tab title** in `xyz.html` should match the demo
  name in the card (emoji + title).

If the demo is a partial / experimental version, mark the card as
`coming-soon` instead of `live` and move it to the second grid.

## 12. PR / branch hygiene

> User's recurring pain point — squash-merges have lost work twice.

- **One demo per PR.** Don't bundle physics tweaks for one demo
  with copy edits for another.
- **Branch from `main`** for each new demo or change. Don't extend a
  previously-merged branch.
- **Memorialize user-blessed constants in memory** so a later branch
  doesn't accidentally revert them.
- After a PR merges, check `main` to confirm the user-blessed values
  survived the squash before starting the next branch.

## 13. Checklist before opening a new demo PR

- [ ] Page renders cleanly in local server preview
- [ ] **Card added to `docs/applications/index.html` "Live demos" grid**
      with the right emoji, n=K, score range, and concise description
- [ ] Browser tab `<title>` matches the index-card name
- [ ] All sliders: every slider's min/max matches decode's range
- [ ] Score panel layout doesn't wrap label lines
- [ ] Reset leaderboard clears every stat row's textContent
- [ ] Search montage updates params on every eval, not just `isNew`
- [ ] Animation token incremented before any new shot starts
- [ ] N_SIM_STEPS large enough for slow-friction physics to come to rest
- [ ] Rest detection confirms across multiple checks
- [ ] "What's happening" copy describes setup, not predictions
- [ ] Save-the-Planet callout present with SKILL.md snippet
- [ ] Smoke-test the simulator with random params before claiming
      anything about the landscape

## 14. User-blessed constants — DO NOT change without an explicit ask

See `memory/slingshot-tuned-params.md` for the slingshot's locked-in
physics. When a user blesses a demo's parameters with "this is
perfect", write a similar memory file before moving on.

## 15. Common feedback flavours (so you can anticipate)

| Feedback | Translation |
|---|---|
| "More power" | Extend force/power upper bound 30–50% |
| "Reduce friction" | Lower friction by 4–10× — be aggressive |
| "Stops too soon" | Increase N_SIM_STEPS + tighten rest threshold |
| "Format is off" | Long string in big-font stat slot → split it |
| "Align with slider" | Decode bounds don't match slider bounds |
| "Reset is broken" | resetEverything() missing a stat row id |
| "Doesn't update" | Param row only updates on isNew, should be every attempt |
| "Looks impossible" | Best-of-2000-random is < 10% of max → tune physics |
| "Looks trivial" | Best-of-2000-random is > 80% of max → tighten scoring |
