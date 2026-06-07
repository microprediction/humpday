"""
FM sound-matching objective: reverse-engineer a synth patch from its spectrum.

A small 4-operator FM voice: a carrier at the fundamental is frequency-modulated
by three modulators locked to harmonics 1/2/3, plus carrier self-feedback. The
HumpDay objective takes a 4-D point in [0,1]^4 — the four "brightness" knobs
(three modulation indices and a feedback amount) — renders the tone, and returns
the **L2 distance between its log-magnitude spectrum and a fixed target patch's**.
Minimising it dials in the target timbre.

Matching a synth to a sound is a real task (recovering a patch you only have a
recording of). Because the harmonic ratios are FIXED, the four knobs shape the
spectrum smoothly, so this version is tractable: CMA-ES, Particle Swarm and
PRIMA_BOBYQA drive the error to near zero. (Free the frequency ratios and it
becomes one of the genuinely hard, octave-trapped inverse problems people throw
global optimisers at.)

Mirrors the browser demo docs/applications/fm-synth.html. Uses numpy.fft.
"""

from __future__ import annotations

import numpy as np

N_DIM = 4
SR = 11025
N = 2048
F0 = 220.0

# Target patch ("Warm" timbre from the browser demo), in [0,1]^4.
TARGET_U = (2 / 6, 1 / 5, 0.3 / 4, 0.10 / 0.85)


def _decode(u):
    return {"I1": u[0] * 6, "I2": u[1] * 5, "I3": u[2] * 4, "fb": u[3] * 0.85}


def render(u, n=N):
    """Render n samples of the steady FM tone for a point in [0,1]^4."""
    p = _decode(u)
    wc = 2 * np.pi * F0 / SR
    out = np.empty(n)
    pc = p1 = p2 = p3 = 0.0
    last = 0.0
    for i in range(n):
        s = np.sin(
            pc
            + p["I1"] * np.sin(p1)
            + p["I2"] * np.sin(p2)
            + p["I3"] * np.sin(p3)
            + p["fb"] * last
        )
        out[i] = s
        last = s
        pc += wc
        p1 += wc
        p2 += 2 * wc
        p3 += 3 * wc
    return out


def log_spectrum(wave):
    """Hann-windowed log-magnitude spectrum (half spectrum)."""
    w = np.hanning(len(wave))
    mag = np.abs(np.fft.rfft(wave * w))
    return np.log1p(mag)


_TARGET_SPEC = log_spectrum(render(TARGET_U))


def spectral_distance(u):
    """RMS distance between this patch's log-spectrum and the target's."""
    spec = log_spectrum(render(u))
    n = min(len(spec), len(_TARGET_SPEC))
    d = spec[1:n] - _TARGET_SPEC[1:n]
    return float(np.sqrt(np.mean(d * d)))


def objective(u):
    """HumpDay objective: spectral distance to the target timbre (minimise)."""
    return spectral_distance(u)
