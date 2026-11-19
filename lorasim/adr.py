"""Network-server ADR loop (the algorithm recommended by Semtech / TTN).

Every node starts at SF12 and 14 dBm. After each block of `history` uplinks
the server looks at the best SNR it heard from the node and computes

    margin = SNR_max - SNR_limit(SF) - margin_db,   steps = floor(margin / 3)

Each positive step lowers the SF by one (down to SF7), then the transmit
power by 3 dB (down to 2 dBm); negative steps raise the power back.
A node the server never hears keeps its settings.
"""
from __future__ import annotations

import numpy as np

from .network import Config, simulate
from .phy import SNR_LIMIT
from .energy import energy_per_uplink

NOISE_DBM = -174 + 10 * np.log10(125e3) + 6.0      # 6 dB gateway noise figure


def adr_update(sf, tx, snr_max, heard, margin_db=10.0):
    sf, tx = sf.copy(), tx.copy()
    lim = np.array([SNR_LIMIT.get(int(s), 0) for s in sf])
    steps = np.floor((snr_max - lim - margin_db) / 3).astype(int)
    for i in np.flatnonzero(heard):
        n = steps[i]
        while n > 0 and sf[i] > 7:
            sf[i] -= 1
            n -= 1
        while n > 0 and tx[i] > 2:
            tx[i] -= 3
            n -= 1
        while n < 0 and tx[i] < 14:
            tx[i] += 3
            n += 1
    return sf, tx


def run_adr(cfg: Config, epochs=10, history=20, margin_db=10.0):
    """Returns per-epoch PDR, SF shares, mean power and energy."""
    n = cfg.n_nodes
    sf = np.full(n, 12)
    tx = np.full(n, 14.0)
    rows = []
    c = Config(**{**cfg.__dict__, "sim_time_s": history * cfg.interval_s})
    for e in range(epochs):
        r = simulate(c, sf_given=sf, tx_given=tx, return_packets=True, traffic_seed=cfg.seed * 1000 + e)
        snr = r["rx_node"] - NOISE_DBM
        heard = r["node_pdr"] > 0
        en = energy_per_uplink(sf, tx, cfg.payload)
        delivered_bits = (r["node_pdr"] * r["node_packets"] * cfg.payload * 8).sum()
        rows.append(dict(epoch=e, pdr=r["pdr"], sf_share=r["sf_share"], mean_tx=float(tx.mean()),
                         energy_mj=float(en.mean() * 1e3),
                         uj_per_bit=float((en * r["node_packets"]).sum() / max(delivered_bits, 1) * 1e6),
                         airtime_load=r["airtime_load"]))
        sf, tx = adr_update(sf, tx, snr, heard, margin_db)
    return rows, sf, tx
