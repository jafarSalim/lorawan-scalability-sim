"""assets/network.gif: two minutes of a 3000-device LoRaWAN cell, sped up.

Every device is a dot; while it transmits it lights up in the colour of its
spreading factor with a ring that grows with time on air. When a packet ends
it flashes white if the gateway got it and red if it was lost (collision or
too weak). Data from lorasim.network.simulate with the smallest-SF policy.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import matplotlib.pyplot as plt  # noqa: E402
import style  # noqa: E402
from lorasim.network import Config, simulate  # noqa: E402


def main(out):
    os.makedirs(out, exist_ok=True)
    cfg = Config(n_nodes=3000, radius_m=4000, interval_s=120.0, sim_time_s=200.0, seed=5)
    r = simulate(cfg, return_packets=True)
    xy, sf, nd = r["xy"] / 1000, r["sf"], r["node"]
    st, en, ok = r["start"], r["end"], r["delivered"]
    style.apply()
    times = np.arange(60, 180, 2.0)
    for f, t in enumerate(times):
        fig = plt.figure(figsize=(10.5, 5.2))
        ax = fig.add_axes([0.01, 0.03, 0.55, 0.9])
        ax.scatter(xy[:, 0], xy[:, 1], s=3, color="#3a2d57")
        act = np.flatnonzero((st <= t) & (en > t))
        for s in range(7, 13):
            a = act[sf[nd[act]] == s]
            if a.size:
                ax.scatter(xy[nd[a], 0], xy[nd[a], 1], s=18 + 6 * (s - 7), color=style.SF_COLORS[s],
                           edgecolors="none")
                frac = np.clip((t - st[a]) / (en[a] - st[a]), 0, 1)
                ax.scatter(xy[nd[a], 0], xy[nd[a], 1], s=60 + 900 * frac, facecolors="none",
                           edgecolors=style.SF_COLORS[s], linewidths=0.8, alpha=0.6)
        done = np.flatnonzero((en > t - 2) & (en <= t))
        good, bad = done[ok[done]], done[~ok[done]]
        ax.scatter(xy[nd[good], 0], xy[nd[good], 1], s=30, color="white", marker="o")
        ax.scatter(xy[nd[bad], 0], xy[nd[bad], 1], s=50, color="#ff3355", marker="x", linewidths=2)
        ax.scatter([0], [0], s=260, marker="^", color="#22d3ee", edgecolors="white", zorder=5)
        ax.set_xlim(-4.2, 4.2)
        ax.set_ylim(-4.2, 4.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"t = {t:5.0f} s   on air now: {act.size}", fontsize=12)
        ax2 = fig.add_axes([0.63, 0.14, 0.34, 0.72])
        fin = en <= t
        tot = max(fin.sum(), 1)
        pdr_sf = [ok[fin & (sf[nd] == s)].mean() if (fin & (sf[nd] == s)).any() else 0 for s in range(7, 13)]
        ax2.bar([f"SF{s}" for s in range(7, 13)], pdr_sf, color=[style.SF_COLORS[s] for s in range(7, 13)])
        ax2.set_ylim(0, 1)
        ax2.set(ylabel="delivered so far", title=f"overall {ok[fin].mean():.2f} of {tot} packets")
        fig.savefig(os.path.join(out, f"f{f:03d}.png"), dpi=80, facecolor=style.BG)
        plt.close(fig)
    print("frames", len(times))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/lora_frames")
