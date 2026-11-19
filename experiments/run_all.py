"""Run every experiment: results/*.json|csv and figures/*.png.

    python experiments/run_all.py                    # everything (about 25 min)
    python experiments/run_all.py phy interference   # only some
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import matplotlib.pyplot as plt  # noqa: E402
import style  # noqa: E402
from lorasim import phy  # noqa: E402
from lorasim.adr import run_adr  # noqa: E402
from lorasim.energy import battery_years, energy_per_uplink  # noqa: E402
from lorasim.network import Config, simulate, path_loss  # noqa: E402

FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
SEEDS = [1, 2, 3]
NODES = [100, 250, 500, 1000, 2000, 4000, 7000, 10000]
BASE = dict(radius_m=4000.0, interval_s=600.0, sim_time_s=2 * 3600.0)
PROCS = int(os.environ.get("PROCS", "2"))


def save_json(name, obj):
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)


def load_json(name):
    with open(os.path.join(RES, name + ".json")) as f:
        return json.load(f)


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)
    print("  figure", name)


# 1 ----------------------------------------------------------------- PHY
def exp_phy():
    rng = np.random.default_rng(1)
    out = {"snr": {}, "sim": {}, "theory": {}, "toa": {}}
    for sf in phy.SFS:
        snr = np.arange(phy.SNR_LIMIT[sf] - 9, phy.SNR_LIMIT[sf] + 1.01, 1.0)
        n = 6000 if sf <= 10 else 2500
        out["snr"][sf] = snr.tolist()
        out["sim"][sf] = [phy.ser_awgn(sf, s, n, rng) for s in snr]
        out["theory"][sf] = [phy.ser_theory(sf, s) for s in snr]
        out["toa"][sf] = {p: phy.time_on_air(sf, p) for p in (10, 20, 51, 115, 222)}
        print(f"  SF{sf} done")
    # SNR where the theory gives SER = 1e-3 (compare with datasheet limits)
    out["snr_at_1e3"] = {}
    for sf in phy.SFS:
        lo, hi = -35.0, 5.0
        for _ in range(50):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if phy.ser_theory(sf, m) > 1e-3 else (lo, m)
        out["snr_at_1e3"][sf] = hi
    save_json("exp1_phy", out)
    with open(os.path.join(RES, "time_on_air.csv"), "w") as f:
        f.write("sf,payload_10B_ms,payload_20B_ms,payload_51B_ms,payload_115B_ms,payload_222B_ms,"
                "datasheet_snr_limit_dB,theory_snr_ser1e-3_dB,sensitivity_dBm\n")
        for sf in phy.SFS:
            f.write(f"{sf}," + ",".join(f"{out['toa'][sf][p] * 1e3:.1f}" for p in (10, 20, 51, 115, 222))
                    + f",{phy.SNR_LIMIT[sf]},{out['snr_at_1e3'][sf]:.2f},{phy.SENSITIVITY[sf]}\n")

    style.apply()
    fig = plt.figure(figsize=(16.5, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.32)
    # (a) chirps: instantaneous frequency of 4 SF7 symbols, and a dechirped spectrum
    ax = fig.add_subplot(gs[0])
    sf = 7
    syms = [0, 32, 96, 64]
    x = phy.modulate(sf, syms)
    f_inst = np.angle(x[1:] * np.conj(x[:-1])) / (2 * np.pi) * 125
    t = np.arange(f_inst.size) / 125e3 * 1e3
    ax.scatter(t, f_inst, s=3, c=f_inst, cmap="cool")
    for k, s in enumerate(syms):
        ax.text((k + 0.5) * 2 ** sf / 125, 70, f"symbol {s}", ha="center", color=style.FG, fontsize=9)
        ax.axvline(k * 2 ** sf / 125, color=style.GRID, lw=1)
    ax.set(xlabel="time (ms)", ylabel="frequency offset (kHz)", ylim=(-70, 80),
           title="SF7: the symbol is where the sweep starts")
    # (b) SER vs SNR
    ax = fig.add_subplot(gs[1])
    for sf in phy.SFS:
        c = style.SF_COLORS[sf]
        snr = np.array(out["snr"][sf])
        ax.semilogy(snr, np.maximum(out["theory"][sf], 1e-6), color=c, lw=1.8)
        sim = np.array(out["sim"][sf])
        ok = sim > 0
        ax.semilogy(snr[ok], sim[ok], "o", color=c, ms=5, mec="white", mew=0.5, label=f"SF{sf}")
        ax.axvline(phy.SNR_LIMIT[sf], color=c, ls=":", lw=1)
    ax.set(xlabel="SNR in 125 kHz (dB)", ylabel="symbol error rate", ylim=(1e-4, 1),
           title="SER: simulation (o) vs theory (—)")
    ax.legend(ncol=2, fontsize=9, loc="lower left")
    ax.text(-6.3, 0.35, "dotted:\ndatasheet\nlimits", color=style.MUTED, fontsize=8.5, ha="right")
    # (c) time on air
    ax = fig.add_subplot(gs[2])
    toa = [out["toa"][sf][20] * 1e3 for sf in phy.SFS]
    bars = ax.bar([f"SF{s}" for s in phy.SFS], toa, color=[style.SF_COLORS[s] for s in phy.SFS])
    for b, v in zip(bars, toa):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.0f} ms", ha="center", fontsize=9.5)
    ax.set_yscale("log")
    ax.set(ylabel="time on air, 20-byte payload (ms)", ylim=(20, 4000),
           title="+1 SF ≈ 2× airtime")
    style.title(fig, "LoRa physical layer, built from scratch",
                "chirp spread spectrum at one sample per chip, 125 kHz; time on air from Semtech AN1200.13")
    savefig(fig, "fig1_phy")


# 2 ------------------------------------------------------- interference
def _sir_row(sf):
    rng = np.random.default_rng(100 + sf)
    row, curves = [], {}
    n_sym = 600 if sf <= 10 else 300
    grid = np.arange(6, -46, -3.0)
    for sfi in phy.SFS:
        sers = []
        for s in grid:
            v = phy.ser_with_interferer(sf, sfi, s, n_sym, rng)
            sers.append(v)
            if v > 0.3:
                break
        sers = np.array(sers)
        g = grid[:sers.size]
        # first SIR (going down) where SER exceeds 1 %: interpolate in log SER
        k = np.flatnonzero(sers > 0.01)
        if k.size == 0:
            thr = float(g[-1])
        elif k[0] == 0:
            thr = float(g[0])
        else:
            i = k[0]
            y0, y1 = np.log10(max(sers[i - 1], 1e-4)), np.log10(sers[i])
            thr = float(g[i - 1] + (np.log10(0.01) - y0) * (g[i] - g[i - 1]) / (y1 - y0))
        row.append(round(thr, 1))
        curves[sfi] = [g.tolist(), sers.tolist()]
    return sf, row, curves


def exp_interference():
    t0 = time.time()
    with Pool(PROCS) as pool:
        res = pool.map(_sir_row, list(reversed(phy.SFS)))
    res.sort()
    M = np.array([r[1] for r in res])
    save_json("exp2_interference", dict(matrix=M.tolist(), curves={r[0]: r[2] for r in res},
                                        definition="SIR (dB) below which the symbol error rate exceeds 1 %"))
    np.savetxt(os.path.join(RES, "sir_matrix.csv"), M, delimiter=",", fmt="%.1f",
               header="rows: wanted SF7..SF12, columns: interfering SF7..SF12; SIR (dB) for SER = 1 %")
    print(f"  matrix in {time.time() - t0:.0f} s\n", M)
    fig_interference()


def fig_interference():
    d = load_json("exp2_interference")
    M = np.array(d["matrix"])
    style.apply()
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.35))
    im = ax[0].imshow(M, cmap="magma_r", vmin=-40, vmax=5)
    for i in range(6):
        for j in range(6):
            ax[0].text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=11, fontweight="bold",
                       color="black" if M[i, j] > -20 else "white")
    ax[0].set_xticks(range(6), [f"SF{s}" for s in phy.SFS])
    ax[0].set_yticks(range(6), [f"SF{s}" for s in phy.SFS])
    ax[0].set(xlabel="interferer", ylabel="wanted signal", title="SIR (dB) the receiver needs (SER < 1 %)")
    ax[0].grid(False)
    cb = fig.colorbar(im, ax=ax[0], fraction=0.046)
    cb.set_label("dB")
    for sf in (7, 10, 12):
        pairs = [(7, "-"), (12, "--")] + ([(sf, ":")] if sf not in (7, 12) else [])
        for sfi, ls in pairs:
            g, s = d["curves"][str(sf)][str(sfi)]
            s = np.maximum(s, 1e-3)
            ax[1].semilogy(g, s, ls, color=style.SF_COLORS[sf], marker="o", ms=3,
                           label=f"SF{sf} hit by SF{sfi}")
    ax[1].axhline(0.01, color=style.MUTED, lw=1)
    ax[1].set(xlabel="signal-to-interference ratio (dB)", ylabel="symbol error rate (zero shown as 1e-3)",
              title="Same SF (receiver already locked): ~0 dB. Other SFs: far lower",
              xlim=(-46, 7), ylim=(8e-4, 1.1))
    ax[1].legend(fontsize=8.5, ncol=3, loc="lower left")
    style.title(fig, "Spreading factors are only quasi-orthogonal",
                "measured with our own CSS modem: asynchronous interferer, random symbols, phase and "
                "a small frequency offset; no noise")
    savefig(fig, "fig2_interference")


# 3 --------------------------------------------------------- scalability
def _sim(args):
    kw, seed = args
    cfg = Config(**{**BASE, **kw, "seed": seed})
    r = simulate(cfg)
    return kw, seed, r["pdr"], r["out_of_range"], r["airtime_load"], r["sf_share"], r["pdr_sf"]


def _sweep(variants):
    jobs = [(dict(v, n_nodes=n), s) for v in variants for n in NODES for s in SEEDS]
    with Pool(PROCS) as pool:
        res = pool.map(_sim, jobs, chunksize=2)
    return res


def _agg(res, key):
    out = {}
    for kw, seed, pdr, oor, load, share, pdr_sf in res:
        k = kw[key] if not callable(key) else key(kw)
        out.setdefault(str(k), {}).setdefault(kw["n_nodes"], []).append(
            dict(pdr=pdr, oor=oor, load=load, share={str(a): b for a, b in share.items()},
                 pdr_sf={str(a): b for a, b in pdr_sf.items()}))
    return out


def exp_scalability():
    pols = ["SF7", "SF12", "min", "random", "equal-airtime"]
    res = _sweep([dict(sf_policy=p) for p in pols])
    out = _agg(res, "sf_policy")
    save_json("exp3_scalability", out)
    rows = ["policy,nodes,pdr_mean,pdr_min,pdr_max,out_of_range,channel_load_erlang"]
    for p in pols:
        for n in NODES:
            v = out[p][n] if n in out[p] else out[p][str(n)]
            pd = [x["pdr"] for x in v]
            rows.append(f"{p},{n},{np.mean(pd):.4f},{min(pd):.4f},{max(pd):.4f},"
                        f"{np.mean([x['oor'] for x in v]):.4f},{np.mean([x['load'] for x in v]):.4f}")
    with open(os.path.join(RES, "exp3_scalability.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    fig_scalability()


def _get(d, n):
    return d[str(n)] if str(n) in d else d[n]


def fig_scalability():
    out = load_json("exp3_scalability")
    style.apply()
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.1))
    for p in ["SF7", "SF12", "random", "min", "equal-airtime"]:
        m = [np.mean([x["pdr"] for x in _get(out[p], n)]) for n in NODES]
        lo = [min(x["pdr"] for x in _get(out[p], n)) for n in NODES]
        hi = [max(x["pdr"] for x in _get(out[p], n)) for n in NODES]
        ax[0].plot(NODES, m, marker="o" if p != "equal-airtime" else "x", color=style.POLICY_COLORS[p],
                   label=style.POLICY_LABEL[p] + (" (= static ADR here)" if p == "equal-airtime" else ""),
                   lw=3 if p == "min" else 2, ls="--" if p == "equal-airtime" else "-", mec="white", mew=0.5)
        ax[0].fill_between(NODES, lo, hi, color=style.POLICY_COLORS[p], alpha=0.15)
    ax[0].set_xscale("log")
    ax[0].set(xlabel="end devices on one gateway", ylabel="packet delivery ratio",
              title="One gateway, 4 km radius, one 20-byte uplink per 10 min", ylim=(0, 1))
    ax[0].legend(fontsize=9.5, loc="lower left")
    # per-SF delivery for the min policy at 4000 nodes, and SF shares
    v = _get(out["min"], 4000)
    sfs = phy.SFS
    share = [np.mean([x["share"][str(s)] for x in v]) for s in sfs]
    pdr_sf = [np.mean([x["pdr_sf"].get(str(s), np.nan) for x in v]) for s in sfs]
    b = ax[1].bar([f"SF{s}" for s in sfs], pdr_sf, color=[style.SF_COLORS[s] for s in sfs])
    for bi, sh, pv in zip(b, share, pdr_sf):
        ax[1].text(bi.get_x() + bi.get_width() / 2, pv + 0.02, f"{pv:.2f}", ha="center", fontsize=10.5,
                   fontweight="bold")
        ax[1].text(bi.get_x() + bi.get_width() / 2, 0.04, f"{100 * sh:.0f} %\nof\nnodes", ha="center",
                   fontsize=9, color="black")
    ax[1].set(ylabel="packet delivery ratio", ylim=(0, 1.12),
              title="Static ADR at 4000 nodes: far nodes pay twice")
    style.title(fig, "How many devices can one gateway serve?",
                "Oulu path-loss model (σ = 7.8 dB shadowing), 3 EU868 channels, 1 % duty cycle, capture and "
                "inter-SF interference on; 3 seeds (band = min–max)")
    savefig(fig, "fig3_scalability")


# 4 ------------------------------------------------- what the model needs
def exp_capture():
    variants = [dict(capture=False, inter_sf=False), dict(capture=True, inter_sf=False),
                dict(capture=True, inter_sf=True), dict(capture=True, inter_sf=True, demod_paths=8)]
    names = ["pure ALOHA, orthogonal SFs", "+ capture effect", "+ inter-SF interference",
             "+ 8 demodulators (SX1301)"]
    res = _sweep([dict(v, sf_policy="min") for v in variants])
    key = lambda kw: names[[i for i, v in enumerate(variants)  # noqa: E731
                            if all(kw.get(k, Config.__dataclass_fields__[k].default) == v2
                                   for k, v2 in v.items())][-1]]
    out = _agg(res, key)
    # ALOHA theory for the first variant: per SF and channel, P = exp(-2 G)
    cfg = Config(**{**BASE, "n_nodes": 1000, "seed": 1})
    r = simulate(cfg)
    share = r["sf_share"]
    oor = r["out_of_range"]
    th = []
    for n in NODES:
        p = 0.0
        for s in phy.SFS:
            g = n * share[s] * phy.time_on_air(s, 20) / (BASE["interval_s"] * 3)
            p += share[s] * np.exp(-2 * g)
        th.append((1 - oor) * p / max(1e-9, sum(share.values())))
    out["theory"] = th
    save_json("exp4_capture", out)
    style.apply()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    cols = ["#9d8cc0", "#39ff88", "#ff4f79", "#ffe14d"]
    for nm, c in zip(names, cols):
        m = [np.mean([x["pdr"] for x in _get(out[nm], n)]) for n in NODES]
        ax.plot(NODES, m, marker="o", color=c, label=nm, mec="white", mew=0.5)
    ax.plot(NODES, th, "--", color="white", lw=1.4, label="theory: pure ALOHA, $e^{-2G}$ per SF and channel")
    ax.set_xscale("log")
    ax.set(xlabel="end devices", ylabel="packet delivery ratio", ylim=(0, 1),
           title="Smallest-SF allocation, one gateway")
    ax.legend(loc="lower left", fontsize=9.5)
    style.title(fig, "Which physical effects change the answer?")
    savefig(fig, "fig4_capture_model")


# 5 ------------------------------------------------------------ gateways
def _gw_layout(k, r=3000.0):
    if k == 1:
        return ((0.0, 0.0),)
    a = 2 * np.pi * np.arange(k) / k
    rr = r * 0.55
    return tuple((float(rr * np.cos(x)), float(rr * np.sin(x))) for x in a)


def exp_gateways():
    variants = [dict(sf_policy="min", gateways=_gw_layout(k), radius_m=6000.0) for k in (1, 2, 4, 8)]
    res = _sweep(variants)
    out = _agg(res, lambda kw: len(kw["gateways"]))
    save_json("exp5_gateways", out)
    style.apply()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
    cols = {"1": "#ff4f79", "2": "#ff9f1c", "4": "#22d3ee", "8": "#39ff88"}
    for k in ("1", "2", "4", "8"):
        m = [np.mean([x["pdr"] for x in _get(out[k], n)]) for n in NODES]
        ax[0].plot(NODES, m, marker="o", color=cols[k], label=f"{k} gateway{'s' if k != '1' else ''}",
                   mec="white", mew=0.5)
    ax[0].set_xscale("log")
    ax[0].set(xlabel="end devices in a 6 km disc", ylabel="packet delivery ratio", ylim=(0, 1),
              title="More gateways: shorter links, faster SFs, receive diversity")
    ax[0].legend()
    for k in ("1", "8"):
        sh = [np.mean([x["share"][str(s)] for x in _get(out[k], 4000)]) for s in phy.SFS]
        ax[1].bar(np.arange(6) + (-0.2 if k == "1" else 0.2), sh, 0.4, color=cols[k],
                  label=f"{k} gateway{'s' if k != '1' else ''}")
    ax[1].set_xticks(range(6), [f"SF{s}" for s in phy.SFS])
    ax[1].set(ylabel="share of devices", title="SF mix")
    ax[1].legend()
    style.title(fig, "Densifying gateways", "gateways on a ring of 1.65 km radius; same traffic as before")
    savefig(fig, "fig5_gateways")


# 6 ------------------------------------------------------------- ADR
def exp_adr():
    cfg = Config(**{**BASE, "n_nodes": 2000, "seed": 7})
    rows, sf, tx = run_adr(cfg, epochs=8)
    static = simulate(Config(**{**BASE, "n_nodes": 2000, "seed": 7, "sim_time_s": 20 * 600.0}),
                      return_packets=True)
    e_static = energy_per_uplink(static["sf"], 14.0)
    bits = (static["node_pdr"] * static["node_packets"] * 160).sum()
    static_row = dict(pdr=static["pdr"], uj_per_bit=float((e_static * static["node_packets"]).sum() / bits * 1e6),
                      energy_mj=float(e_static.mean() * 1e3))
    # battery life per SF/power at a 10 min interval
    life = {s: {p: float(battery_years(s, p, 600.0)) for p in (2, 8, 14)} for s in phy.SFS}
    save_json("exp6_adr", dict(epochs=rows, static=static_row, life=life,
                               final_sf={int(s): int((sf == s).sum()) for s in phy.SFS},
                               final_tx={int(p): int((tx == p).sum()) for p in np.unique(tx)}))
    style.apply()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.9))
    ep = [r["epoch"] for r in rows]
    bottom = np.zeros(len(rows))
    for s in phy.SFS:
        v = np.array([r["sf_share"][s] for r in rows])
        ax[0].bar(ep, v, bottom=bottom, color=style.SF_COLORS[s], label=f"SF{s}", width=0.8)
        bottom += v
    ax[0].set(xlabel="ADR round (20 uplinks each)", ylabel="share of devices",
              title="ADR moves devices off SF12")
    ax[0].legend(ncol=3, fontsize=8.5, loc="upper right")
    a2 = ax[1]
    a2.plot(ep, [r["pdr"] for r in rows], "o-", color="#22d3ee", label="ADR: delivery ratio")
    a2.axhline(static_row["pdr"], color="#22d3ee", ls=":", lw=1.4, label="static smallest SF (6 dB margin)")
    a2.plot(ep, [r["mean_tx"] / 14 for r in rows], "s-", color="#ff9f1c", label="ADR: mean TX power / 14 dBm")
    a2.set(xlabel="ADR round", ylim=(0, 1.05), title="Delivery and transmit power, 2000 devices")
    a2.legend(fontsize=9, loc="lower right")
    ax[2].plot(ep, [r["uj_per_bit"] for r in rows], "o-", color="#ff4f79", label="ADR")
    ax[2].axhline(static_row["uj_per_bit"], color="#ff4f79", ls=":", label="static smallest SF")
    ax[2].set_yscale("log")
    ax[2].set(xlabel="ADR round", ylabel="µJ per delivered bit", title="Energy per delivered bit")
    ax[2].legend()
    style.title(fig, "Adaptive data rate (network-server algorithm, 10 dB margin)",
                "all devices start at SF12 / 14 dBm; static shadowing, so the SNR the server sees is exact")
    savefig(fig, "fig6_adr")


EXPS = {"phy": exp_phy, "interference": exp_interference, "scalability": exp_scalability,
        "capture": exp_capture, "gateways": exp_gateways, "adr": exp_adr,
        "fig_interference": fig_interference, "fig_scalability": fig_scalability}

if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    for n in sys.argv[1:] or ["phy", "interference", "scalability", "capture", "gateways", "adr"]:
        t = time.time()
        print(n)
        EXPS[n]()
        print(f"  done in {time.time() - t:.0f} s")
