# FM Sound Match — A Spectral Inverse Problem

Reverse-engineer a synth patch from its spectrum. A small 4-operator FM voice
(carrier + three harmonic modulators + feedback) is controlled by HumpDay's
`[0,1]^4` cube — four "brightness" knobs. The objective renders the tone and
returns the **L2 distance between its log-magnitude spectrum and a fixed target
patch's**. Minimising it recovers the target timbre.

## What this stresses

- **An inverse problem.** Matching a synth to a recording is a real task; the
  optimiser only ever sees the spectral distance, never the knobs that made the
  target.
- **A smooth (here) landscape.** Because the harmonic ratios are *fixed*, the
  four knobs shape the spectrum smoothly, so `CMAEvolutionStrategy`,
  `ParticleSwarm` and `PRIMA_BOBYQA` reliably drive the error to near zero while
  `NelderMead`, `Powell` and Random Search settle for a rough likeness.
- **Why it's bounded.** Free the frequency ratios and this becomes one of the
  genuinely hard, *octave-trapped* inverse problems — a good reason the demo
  keeps them fixed.

## Running

```bash
python -m example_applications.fm_sound_match.run
```

Uses `numpy.fft`. Expect the good methods to recover the target knobs
(≈ 2.0 / 1.0 / 0.3 / 0.10). Mirrors the browser demo
[`docs/applications/fm-synth.html`](../../docs/applications/fm-synth.html), which
also *plays* the matched sound.
