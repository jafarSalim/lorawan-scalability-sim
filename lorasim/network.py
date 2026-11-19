"""Discrete-event LoRaWAN uplink simulator.

Nodes are dropped uniformly in a disc around one or more gateways. Each node
sends a 20-byte uplink on one of the three EU868 default channels, at random
(Poisson) times with a mean interval, never faster than the 1 % duty cycle
allows. A packet is received by a gateway when

1. its power is above the gateway sensitivity for its spreading factor,
2. it survives every packet that overlaps it in time on the same channel:
   the signal-to-interference ratio against the summed power of the
   interferers of each spreading factor must exceed the rejection threshold
   of that pair (co-SF capture on the diagonal, inter-SF rejection off it),
3. (optionally) one of the gateway's 8 demodulation paths is free when its
   preamble arrives (SX1301 limit).

A packet is delivered if at least one gateway receives it.

Propagation: log-distance path loss with log-normal shadowing, fitted by
Petajajarvi et al. (ITST 2015) to LoRa measurements at 868 MHz in Oulu
(gateway on a 24 m mast, nodes at ground level):
PL(d) = 128.95 + 23.2 log10(d / 1 km) + X,  X ~ N(0, 7.8 dB).
"""
from __future__ import annotations

import heapq
import os
from dataclasses import dataclass, field

import numpy as np

from .phy import SENSITIVITY, SFS, time_on_air

CHANNELS = 3
PL0, D0, GAMMA, SHADOW = 128.95, 1000.0, 2.32, 7.8

# Rejection thresholds (dB): rows = wanted SF7..12, columns = interfering
# SF7..12. The off-diagonal (inter-SF) values are measured with our own CSS
# modem in experiment 2 (results/sir_matrix.csv). The diagonal (same SF) is
# set to the 6 dB capture threshold measured on real hardware by Bor et al.
# (MSWiM 2016): our modem assumes it is already locked to the wanted packet,
# which makes its own co-SF figure (about -1 to -5 dB) far too optimistic for
# a gateway that must first detect the right preamble.
CO_SF_CAPTURE_DB = 6.0
_SIR_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "sir_matrix.csv")


def default_sir():
    if os.path.exists(_SIR_CSV):
        m = np.loadtxt(_SIR_CSV, delimiter=",")
    else:   # coarse placeholder until experiment 2 has run
        m = np.array([[-(10 + 2.5 * i) for _ in range(6)] for i in range(6)], dtype=float)
    np.fill_diagonal(m, CO_SF_CAPTURE_DB)
    return m


@dataclass
class Config:
    n_nodes: int = 1000
    radius_m: float = 5000.0
    gateways: tuple = ((0.0, 0.0),)
    interval_s: float = 600.0          # mean time between uplinks of one node
    sim_time_s: float = 6 * 3600.0
    payload: int = 20
    tx_dbm: float = 14.0
    sf_policy: str = "min"             # min | SF7 | SF12 | random | equal-airtime | given
    sf_margin_db: float = 6.0          # link margin when choosing the smallest SF
    capture: bool = True               # False: any overlap on the same SF destroys both
    inter_sf: bool = True              # model imperfect orthogonality between SFs
    sir: np.ndarray = field(default_factory=default_sir)
    demod_paths: int = 0               # 0 = unlimited, 8 = SX1301
    seed: int = 0


def path_loss(d, rng=None, shadow=True):
    pl = PL0 + 10 * GAMMA * np.log10(np.maximum(d, 1.0) / D0)
    if shadow and rng is not None:
        pl = pl + rng.normal(0, SHADOW, np.shape(d))
    return pl


def drop_nodes(cfg, rng):
    r = cfg.radius_m * np.sqrt(rng.random(cfg.n_nodes))
    a = 2 * np.pi * rng.random(cfg.n_nodes)
    return np.c_[r * np.cos(a), r * np.sin(a)]


def link_budget(cfg, xy, rng):
    """Received power (dBm) of every node at every gateway, with fixed shadowing."""
    gw = np.asarray(cfg.gateways, float)
    d = np.linalg.norm(xy[:, None, :] - gw[None, :, :], axis=2)
    return cfg.tx_dbm - path_loss(d, rng), d


def assign_sf(cfg, rx_best, rng):
    """Spreading factor per node from the policy."""
    n = rx_best.size
    if cfg.sf_policy == "SF7":
        return np.full(n, 7)
    if cfg.sf_policy == "SF12":
        return np.full(n, 12)
    if cfg.sf_policy == "random":
        return rng.integers(7, 13, n)
    # smallest SF whose sensitivity + margin is met (what converged ADR aims at)
    sf = np.full(n, 12)
    for s in reversed(SFS):
        ok = rx_best >= SENSITIVITY[s] + cfg.sf_margin_db
        sf[ok] = s
    if cfg.sf_policy == "min":
        return sf
    if cfg.sf_policy == "equal-airtime":
        # EXPLoRa-AT idea: fill SFs so that each carries the same total airtime,
        # giving the nearest nodes the fastest SFs, never a SF below the minimum.
        toa = np.array([time_on_air(s, cfg.payload) for s in SFS])
        share = (1 / toa) / (1 / toa).sum()          # node fraction per SF for equal airtime
        order = np.argsort(-rx_best)                 # strongest first
        out = np.empty(n, int)
        start = 0
        for k, s in enumerate(SFS):
            m = int(round(share[k] * n)) if k < len(SFS) - 1 else n - start
            out[order[start:start + m]] = s
            start += m
        return np.maximum(out, sf)
    raise ValueError(cfg.sf_policy)


def generate_traffic(cfg, sf, rng):
    """Packet start times, node ids and channels (Poisson, duty-cycle limited)."""
    toa = np.array([time_on_air(s, cfg.payload) if s >= 7 else 0.0 for s in range(13)])
    starts, nodes = [], []
    for i in range(cfg.n_nodes):
        gap_min = toa[sf[i]] * 99                    # 1 % duty cycle
        t = rng.random() * cfg.interval_s
        m = int(cfg.sim_time_s / cfg.interval_s * 2.5) + 5
        gaps = np.maximum(rng.exponential(cfg.interval_s, m), gap_min)
        ts = t + np.concatenate(([0.0], np.cumsum(gaps)))
        ts = ts[ts < cfg.sim_time_s]
        starts.append(ts)
        nodes.append(np.full(ts.size, i))
    st = np.concatenate(starts)
    nd = np.concatenate(nodes)
    o = np.argsort(st)
    st, nd = st[o], nd[o]
    ch = rng.integers(0, CHANNELS, st.size)
    return st, nd, ch, toa


def simulate(cfg: Config, sf_given=None, tx_given=None, return_packets=False, traffic_seed=None):
    """Run one deployment. sf_given / tx_given override the SF policy and the
    transmit power per node; traffic_seed redraws the traffic but keeps the
    same node positions and shadowing (used by the ADR loop)."""
    rng = np.random.default_rng(cfg.seed)
    xy = drop_nodes(cfg, rng)
    rx, dist = link_budget(cfg, xy, rng)                  # n x G at cfg.tx_dbm
    best = rx.max(axis=1)
    sf = sf_given if sf_given is not None else assign_sf(cfg, best, rng)
    tx = np.full(cfg.n_nodes, cfg.tx_dbm) if tx_given is None else np.asarray(tx_given, float)
    rx = rx - cfg.tx_dbm + tx[:, None]
    if traffic_seed is not None:
        rng = np.random.default_rng(traffic_seed)
    st, nd, ch, toa = generate_traffic(cfg, sf, rng)
    en = st + toa[sf[nd]]
    psf = sf[nd]
    G = rx.shape[1]
    prx = rx[nd]                                          # P x G dBm
    plin = 10 ** (prx / 10)
    P = st.size
    ok = np.zeros((P, G), bool)
    sens = np.array([SENSITIVITY.get(s, 0) for s in range(13)])
    above = prx >= sens[psf][:, None]
    maxtoa = toa[12]
    sir = cfg.sir
    # interference check, channel by channel, with a sliding window on start times
    for c in range(CHANNELS):
        idx = np.flatnonzero(ch == c)
        s_c, e_c = st[idx], en[idx]
        for a, i in enumerate(idx):
            if not above[i].any():
                continue
            lo = np.searchsorted(s_c, s_c[a] - maxtoa, "left")
            hi = np.searchsorted(s_c, e_c[a], "left")
            cand = np.arange(lo, hi)
            cand = cand[(cand != a) & (e_c[cand] > s_c[a])]
            if cand.size == 0:
                ok[i] = above[i]
                continue
            j = idx[cand]
            good = above[i].copy()
            for s in np.unique(psf[j]):
                js = j[psf[j] == s]
                if s == psf[i]:
                    if not cfg.capture:
                        good[:] = False
                        break
                    thr = sir[psf[i] - 7, s - 7]
                elif cfg.inter_sf:
                    thr = sir[psf[i] - 7, s - 7]
                else:
                    continue
                interf = plin[js].sum(axis=0)
                good &= 10 * np.log10(plin[i] / interf) >= thr
            ok[i] = good
    if cfg.demod_paths:
        # each gateway can lock at most `demod_paths` packets at once; a packet
        # needs a free path when its preamble starts (first come, first served)
        for g in range(G):
            cand = np.flatnonzero(above[:, g])
            busy = []                                     # end times of occupied paths
            for i in cand[np.argsort(st[cand])]:
                while busy and busy[0] <= st[i]:
                    heapq.heappop(busy)
                if len(busy) < cfg.demod_paths:
                    heapq.heappush(busy, en[i])
                else:
                    ok[i, g] = False
    delivered = ok.any(axis=1)
    out = dict(pdr=float(delivered.mean()) if P else 1.0, packets=P,
               pdr_sf={int(s): float(delivered[psf == s].mean()) for s in np.unique(psf)},
               sf_share={int(s): float((sf == s).mean()) for s in SFS},
               out_of_range=float((~above.any(axis=1)).mean()),
               airtime_load=float((en - st).sum() / (cfg.sim_time_s * CHANNELS)),
               delivered=delivered, node=nd, sf=sf, xy=xy, best=best, start=st, end=en, ch=ch,
               dist=dist.min(axis=1), prx=prx, tx=tx, rx_node=rx.max(axis=1),
               node_pdr=np.bincount(nd, delivered, cfg.n_nodes) / np.maximum(np.bincount(nd, minlength=cfg.n_nodes), 1),
               node_packets=np.bincount(nd, minlength=cfg.n_nodes))
    if not return_packets:
        for k in ("delivered", "node", "start", "end", "ch", "prx"):
            out.pop(k)
    return out
