import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lorasim import phy  # noqa: E402
from lorasim.adr import adr_update  # noqa: E402
from lorasim.energy import energy_per_uplink  # noqa: E402
from lorasim.network import Config, simulate  # noqa: E402


def test_modulate_demodulate_noiseless():
    rng = np.random.default_rng(0)
    for sf in phy.SFS:
        s = rng.integers(0, 2 ** sf, 50)
        assert (phy.demodulate(sf, phy.modulate(sf, s)) == s).all()


def test_time_on_air_matches_semtech_calculator():
    # Semtech LoRa calculator, 20-byte payload, 125 kHz, CR 4/5, 8-symbol preamble, CRC on
    assert abs(phy.time_on_air(7, 20) * 1e3 - 56.6) < 0.1
    assert abs(phy.time_on_air(12, 20) * 1e3 - 1318.9) < 0.5


def test_ser_simulation_matches_theory():
    rng = np.random.default_rng(1)
    sim = phy.ser_awgn(8, -12.0, 8000, rng)
    th = phy.ser_theory(8, -12.0)
    assert abs(sim - th) < 4 * np.sqrt(th / 8000) + 1e-3


def test_more_nodes_lower_delivery():
    a = simulate(Config(n_nodes=200, sim_time_s=3600, seed=1))["pdr"]
    b = simulate(Config(n_nodes=5000, sim_time_s=3600, seed=1))["pdr"]
    assert a > b


def test_no_capture_is_worse():
    c1 = simulate(Config(n_nodes=3000, sim_time_s=3600, seed=2))["pdr"]
    c0 = simulate(Config(n_nodes=3000, sim_time_s=3600, seed=2, capture=False))["pdr"]
    assert c1 >= c0


def test_adr_lowers_sf_for_strong_links():
    sf, tx = adr_update(np.array([12, 12]), np.array([14.0, 14.0]), np.array([10.0, -25.0]),
                        np.array([True, True]))
    assert sf[0] == 7 and sf[1] == 12


def test_energy_grows_with_sf():
    e = energy_per_uplink(np.array([7, 12]), 14.0)
    assert e[1] > 10 * e[0]
