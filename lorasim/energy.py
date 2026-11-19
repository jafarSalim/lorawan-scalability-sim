"""Energy per uplink of a class-A LoRaWAN end device.

Transmit current follows a simple PA model calibrated on the SX1276 datasheet
(PA_BOOST, 87 mA at +17 dBm): I_tx = I_0 + P_out / (eta * V) with
I_0 = 11 mA (radio core + MCU), eta = 0.2 (PA efficiency), V = 3.3 V.
After every uplink the device opens two receive windows (RX1 at the uplink
SF, RX2 at SF12) and, with no downlink, listens about 6 symbols in each
before going back to sleep.
"""
from __future__ import annotations

import numpy as np

from .phy import time_on_air

V = 3.3
I0 = 11e-3
ETA = 0.2
I_RX = 11e-3
I_SLEEP = 1.5e-6
RX_SYMBOLS = 6


def tx_current(dbm):
    p = 10 ** ((np.asarray(dbm, float) - 30) / 10)
    return I0 + p / (ETA * V)


def energy_per_uplink(sf, dbm, payload=20, bw=125e3):
    sf = np.asarray(sf)
    toa = np.vectorize(lambda s: time_on_air(int(s), payload))(sf)
    t_rx = RX_SYMBOLS * (2.0 ** sf / bw) + RX_SYMBOLS * (2.0 ** 12 / bw)
    return V * (tx_current(dbm) * toa + I_RX * t_rx)        # joules


def battery_years(sf, dbm, interval_s, capacity_mah=2400, payload=20):
    e = energy_per_uplink(sf, dbm, payload)
    i_avg = e / V / interval_s + I_SLEEP
    return capacity_mah * 1e-3 / i_avg / (365.25 * 24)          # Ah / A = hours
