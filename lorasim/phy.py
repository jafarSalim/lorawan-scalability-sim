"""LoRa physical layer from scratch: chirp spread spectrum (CSS).

A LoRa symbol with spreading factor SF carries SF bits as one of 2^SF cyclic
shifts of an up-chirp that sweeps the bandwidth B in 2^SF / B seconds. The
receiver multiplies by a down-chirp ("dechirp") and takes an FFT: the shift
shows up as a single tone, so demodulation is an argmax over 2^SF bins.

Everything here is at one sample per chip (fs = B), the usual baseband model.
"""
from __future__ import annotations

import math

import numpy as np

SFS = (7, 8, 9, 10, 11, 12)


def upchirp(sf, shift=0):
    """Base-band up-chirp of 2^sf chips, cyclically shifted by `shift` chips."""
    n = 2 ** sf
    k = (np.arange(n) + shift) % n
    return np.exp(1j * np.pi * k * k / n - 1j * np.pi * k)


def modulate(sf, symbols):
    """Concatenate symbols (ints in [0, 2^sf)) into one complex baseband signal."""
    n = 2 ** sf
    k = (np.arange(n)[None, :] + np.asarray(symbols)[:, None]) % n
    return np.exp(1j * np.pi * k * k / n - 1j * np.pi * k).reshape(-1)


def demodulate(sf, x):
    """Dechirp + FFT demodulator; x length must be a multiple of 2^sf."""
    n = 2 ** sf
    down = np.conj(upchirp(sf))
    y = x.reshape(-1, n) * down[None, :]
    return np.argmax(np.abs(np.fft.fft(y, axis=1)), axis=1)


def ser_awgn(sf, snr_db, n_sym=20000, rng=None, batch=4000):
    """Monte Carlo symbol error rate in AWGN; SNR measured over the bandwidth."""
    rng = np.random.default_rng() if rng is None else rng
    n = 2 ** sf
    err = 0
    done = 0
    sigma = 10 ** (-snr_db / 20) / math.sqrt(2)
    while done < n_sym:
        b = min(batch, n_sym - done)
        s = rng.integers(0, n, b)
        x = modulate(sf, s)
        x = x + sigma * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
        err += int((demodulate(sf, x) != s).sum())
        done += b
    return err / n_sym


def ser_theory(sf, snr_db):
    """Exact SER of non-coherent orthogonal signalling with M = 2^sf tones.

    After dechirping, the FFT bin of the sent symbol is Rician with energy
    Es/N0 = 2^sf * SNR, the other M - 1 bins are Rayleigh; the symbol is
    right when the signal bin is the largest:
        P_c = int f_Rice(r) (1 - exp(-r^2))^(M-1) dr
    """
    M = 2 ** sf
    es = (10 ** (snr_db / 10)) * M
    r = np.linspace(0, math.sqrt(es) + 12, 4000)
    from scipy.special import i0e
    pdf = 2 * r * np.exp(-(r - math.sqrt(es)) ** 2) * i0e(2 * r * math.sqrt(es))   # Rician, unit noise
    cdf_noise = 1 - np.exp(-r ** 2)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return max(1 - trap(pdf * cdf_noise ** (M - 1), r), 0.0)


def ser_with_interferer(sf, sf_i, sir_db, n_sym=2000, rng=None):
    """SER of an SF signal hit by an asynchronous interferer of spreading factor sf_i.

    The interferer is a random symbol stream, with random timing and carrier
    phase, scaled to the given signal-to-interference ratio. No noise.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = 2 ** sf
    s = rng.integers(0, n, n_sym)
    x = modulate(sf, s)
    ni = 2 ** sf_i
    n_int = int(math.ceil(x.size / ni)) + 2
    xi = modulate(sf_i, rng.integers(0, ni, n_int))
    off = rng.integers(0, ni)
    xi = xi[off:off + x.size]
    g = 10 ** (-sir_db / 20) * np.exp(2j * np.pi * rng.random())
    # a small carrier offset between the two transmitters (random within +/- 0.3 bin)
    cfo = (rng.random() - 0.5) * 0.6 / n
    xi = xi * np.exp(2j * np.pi * cfo * np.arange(x.size))
    return float((demodulate(sf, x + g * xi) != s).mean())


# ---------------------------------------------------------------- airtime
def time_on_air(sf, payload=20, bw=125e3, cr=1, preamble=8, header=True, crc=True, ldro=None):
    """Semtech SX127x time-on-air (AN1200.13) in seconds."""
    tsym = 2 ** sf / bw
    if ldro is None:
        ldro = tsym > 16e-3
    de = 1 if ldro else 0
    ih = 0 if header else 1
    num = 8 * payload - 4 * sf + 28 + 16 * int(crc) - 20 * ih
    n_payload = 8 + max(math.ceil(num / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (preamble + 4.25) * tsym + n_payload * tsym


# SX1276 sensitivity at 125 kHz (datasheet, dBm) and demodulation SNR limits (dB)
SENSITIVITY = {7: -123.0, 8: -126.0, 9: -129.0, 10: -132.0, 11: -134.5, 12: -137.0}
SNR_LIMIT = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}
