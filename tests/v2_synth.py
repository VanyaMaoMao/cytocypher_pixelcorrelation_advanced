"""Synthetic PixelCorrelation-like waveforms for v2 tests (not a test module)."""
import numpy as np

FS = 250.0


def gauss(t, c, sigma, amp=1.0):
    return amp * np.exp(-0.5 * ((t - c) / sigma) ** 2)


def make_amplitude(centers, amps=None, sigma=0.04, dur=6.0, fs=FS, noise=0.003, seed=0, extra=None):
    """Baseline-relative contraction amplitude (contractions upward, rest 0)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fs)) / fs
    a = np.zeros_like(t)
    amps = amps if amps is not None else [1.0] * len(centers)
    for c, h in zip(centers, amps):
        a += gauss(t, c, sigma, h)
    if extra is not None:
        a += extra(t)
    a += rng.normal(0, noise, t.size)
    return t, a


def to_raw(amplitude, polarity, offset=0.3, ceiling=0.95):
    """Raw PixelCorrelation with given physiological polarity (+1 up / -1 down)."""
    return offset + amplitude if polarity > 0 else ceiling - amplitude
