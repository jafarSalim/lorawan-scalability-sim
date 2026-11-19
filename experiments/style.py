"""Figure style: "night radio" — phosphor colours on near-black violet."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG = "#0d0717"
PANEL = "#160d26"
FG = "#f1eafe"
GRID = "#33244f"
MUTED = "#9d8cc0"
SF_COLORS = {7: "#39ff88", 8: "#b4ff39", 9: "#ffe14d", 10: "#ff9f1c", 11: "#ff4f79", 12: "#c86bff"}
POLICY_COLORS = {"SF7": "#39ff88", "SF12": "#c86bff", "min": "#22d3ee", "random": "#ffe14d",
                 "equal-airtime": "#ff4f79", "ADR": "#ff9f1c"}
POLICY_LABEL = {"SF7": "all SF7", "SF12": "all SF12", "min": "smallest SF that reaches (static ADR)",
                "random": "random SF", "equal-airtime": "equal airtime per SF"}


def apply():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": FG, "axes.titlecolor": FG,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": FG,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
        "legend.facecolor": PANEL, "legend.edgecolor": GRID, "legend.framealpha": 0.92,
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12.5,
        "axes.titleweight": "bold", "axes.labelsize": 11, "lines.linewidth": 2.2,
        "savefig.dpi": 170, "savefig.bbox": "tight",
    })


def title(fig, text, sub=None):
    fig.suptitle(text, x=0.01, ha="left", fontsize=15, fontweight="bold", color=FG, y=1.03)
    if sub:
        fig.text(0.01, 0.985, sub, ha="left", va="top", fontsize=10.5, color=MUTED)
