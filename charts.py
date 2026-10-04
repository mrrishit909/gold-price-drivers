"""Step 6: charts.   ./venv/bin/python charts.py -> charts/00_cover.png, 01_prices.png, 02_drivers.png, 03_rolling.png, 04_forecast.png, 05_scatter.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import DRIVERS, LAST, fred, gold

HERE = Path(__file__).parent
R = HERE / "results"
INK, DIM, GRID, BG, BLUE, ORANGE, GRAY, GREEN = "#f2f2f0", "#8a8a87", "#1d1d1d", "#0b0b0b", "#3987e5", "#d95926", "#9a9a96", "#3aa876"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "axes.edgecolor": "#3a3a3a", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
                     "axes.titlelocation": "left", "axes.titlepad": 12})
LR = pd.read_csv(R / "longrun.csv").set_index("key")["value"]; DEC = pd.read_csv(R / "decades.csv"); RG = pd.read_csv(R / "regress.csv"); CH = pd.read_csv(R / "chow.csv").iloc[0]; RO = pd.read_csv(R / "rolling.csv"); SI = pd.read_csv(R / "simple.csv").set_index("sample"); FC = pd.read_csv(R / "forecast.csv"); AC = pd.read_csv(R / "acf.csv"); SU = pd.read_csv(R / "summary.csv").set_index("key")["value"]
g = gold(); series = {sid: fred(sid)[0] for sid in ("DFII10", "T10YIE", "DTWEXBGS", "VIXCLS", "CPIAUCSL")}; idx = pd.period_range("1960-01", LAST, freq="M"); cpi = series["CPIAUCSL"].reindex(idx); gl = g.reindex(idx); real = gl * cpi.iloc[-1] / cpi
X = lambda p: p.to_timestamp()
NAMES = {"d_real_yield": "real yield", "d_dollar": "dollar", "d_breakeven": "breakeven\ninflation", "d_vix": "VIX"}


def save(fig, name):
    fig.tight_layout(); fig.savefig(HERE / "charts" / name, dpi=150); plt.close(fig)


def prices(ax, fs=9):
    ax.plot(X(idx), gl.values, color=GRAY, linewidth=1.4, label="nominal dollars"); ax.plot(X(idx), real.values, color=BLUE, linewidth=2.0, label="August 2026 dollars")
    pk = pd.Period(str(int(LR["real peak month before 2000"]))[:4] + "-" + str(int(LR["real peak month before 2000"]))[4:], "M"); ax.scatter([X(pd.PeriodIndex([pk]))[0]], [LR["real peak before 2000 (August 2026 dollars)"]], color=ORANGE, s=44, zorder=4)
    ax.annotate(f"January 1980: ${LR['real peak before 2000 (August 2026 dollars)']:,.0f}", (X(pd.PeriodIndex([pk]))[0], LR["real peak before 2000 (August 2026 dollars)"]), xytext=(15, 8), textcoords="offset points", color=INK, fontsize=fs)
    ax.axhline(LR["real peak before 2000 (August 2026 dollars)"], color=ORANGE, linestyle=":", linewidth=1); ax.set_ylim(0, 5400); ax.set_ylabel("dollars per troy ounce"); ax.legend(frameon=False, labelcolor=INK, fontsize=fs, loc="upper left"); ax.set_title("The price of gold, nominal and in today's dollars", fontsize=12)


def decades(ax):
    x = np.arange(len(DEC)); w = 0.38; ax.bar(x - w / 2, 100 * DEC["nominal_cagr"], width=w * 0.92, color=GRAY, label="nominal"); ax.bar(x + w / 2, 100 * DEC["real_cagr"], width=w * 0.92, color=BLUE, label="after inflation")
    for xi, v in list(zip(x - w / 2, 100 * DEC["nominal_cagr"])) + list(zip(x + w / 2, 100 * DEC["real_cagr"])):
        ax.text(xi, v + (0.8 if v >= 0 else -0.8), f"{v:.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
    ax.axhline(0, color=DIM, linewidth=0.9); ax.set_xticks(x); ax.set_xticklabels(DEC["decade"]); ax.set_ylabel("% a year, start to end of decade"); ax.set_ylim(-15, 34); ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right"); ax.grid(axis="x", visible=False); ax.set_title("Gold, decade by decade", fontsize=12)


def drivers(axes):
    ax = axes[0]; x = np.arange(4); w = 0.38
    for dx, name, c, lab in ((-w / 2, "2006-02 to 2021-12", GRAY, "2006 to 2021"), (w / 2, "2022-01 to 2026-08", BLUE, "2022 to 2026")):
        d = RG[RG["sample"] == name].set_index("term"); v = [d.loc[t, "coefficient"] * SU[f"sd_{t}"] for t in DRIVERS]; e = [1.96 * d.loc[t, "se"] * SU[f"sd_{t}"] for t in DRIVERS]
        ax.bar(x + dx, v, width=w * 0.92, color=c, label=lab, yerr=e, error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
    ax.axhline(0, color=DIM, linewidth=0.9); ax.set_xticks(x); ax.set_xticklabels([NAMES[t] for t in DRIVERS], fontsize=9); ax.set_ylabel("effect on gold's monthly return, % per 1 SD move"); ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower right"); ax.grid(axis="x", visible=False); ax.set_title("What a typical move in each driver does", fontsize=12)
    ax = axes[1]; d = RG[RG["term"] == "intercept"]; r2 = RG.drop_duplicates("sample").set_index("sample")["r2"]; labs = ["2006 to 2021", "2022 to 2026"]; vals = [r2["2006-02 to 2021-12"], r2["2022-01 to 2026-08"]]
    ax.bar([0, 1], [100 * v for v in vals], color=[GRAY, BLUE], width=0.55); [ax.text(i, 100 * v + 1, f"{100 * v:.0f}%", ha="center", fontsize=10, color=INK) for i, v in enumerate(vals)]
    inter = [d[d["sample"] == s]["coefficient"].iloc[0] for s in ("2006-02 to 2021-12", "2022-01 to 2026-08")]
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"{l}\nbaseline {c:+.1f}% a month" for l, c in zip(labs, inter)], fontsize=9); ax.set_ylim(0, 60); ax.set_ylabel("share of monthly variation explained (R²)"); ax.grid(axis="x", visible=False); ax.set_title(f"Same drivers, different gold (Chow p = {CH['p_value']:.4f})", fontsize=12)


def rolling(axes):
    x = pd.PeriodIndex(RO["end_month"], freq="M").to_timestamp()
    ax = axes[0]; ax.plot(x, RO["slope_real_yield_alone"], color=BLUE, linewidth=2.0, label="real yield alone"); ax.plot(x, RO["coef_real_yield"], color=GRAY, linewidth=1.6, linestyle="--", label="with dollar, breakeven, VIX held fixed"); ax.axhline(0, color=DIM, linewidth=0.9); ax.axvline(pd.Timestamp("2022-01-01"), color=ORANGE, linestyle=":", linewidth=1.2)
    ax.set_ylabel("return per 1-point rise in real yield (%)"); ax.set_title("The real-yield slope, 60 months at a time", fontsize=12); ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="upper left")
    ax = axes[1]; ax.plot(x, 100 * RO["r2_real_yield_alone"], color=BLUE, linewidth=2.0, label="real yield alone"); ax.plot(x, 100 * RO["r2"], color=GREEN, linewidth=2.0, label="all four drivers"); ax.axvline(pd.Timestamp("2022-01-01"), color=ORANGE, linestyle=":", linewidth=1.2); ax.set_ylabel("R² (%)"); ax.set_ylim(0, 70); ax.set_title("How much they explain, 60 months at a time", fontsize=12); ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="lower left")


def forecast(axes):
    ax = axes[0]; d = FC.copy(); order = ["AR(1)", "momentum", "drivers, lagged", "drivers, same month (oracle)"]; d = d.set_index("model").loc[order]; labs = ["last month's\nreturn", "past year's\nreturn", "last month's\ndrivers", "this month's\ndrivers\n(not knowable)"]
    cols = [GRAY, GRAY, BLUE, ORANGE]; ax.bar(range(4), 100 * d["r2_oos"], color=cols, width=0.6)
    for i, (v, p) in enumerate(zip(d["r2_oos"], d["dm_p"])): ax.text(i, 100 * v + (1 if v >= 0 else -1), f"{100 * v:.1f}%\np = {p:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
    ax.axhline(0, color=DIM, linewidth=0.9); ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8.5); ax.set_ylim(-6, 32); ax.set_ylabel("out-of-sample R² against the average return (%)"); ax.grid(axis="x", visible=False); ax.set_title("Forecasting next month's return", fontsize=12)
    ax = axes[1]; a = AC[AC["lag"] == 1]; labs = ["1975 to 2026", "2006 to 2026", "averaging\nalone"]; vals = list(a["autocorrelation"]); ax.bar(range(3), vals, color=[GRAY, GRAY, ORANGE], width=0.55)
    for i, v in enumerate(vals): ax.text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=9); ax.set_ylim(0, 0.32); ax.set_ylabel("correlation of a month's return with the last month's"); ax.grid(axis="x", visible=False); ax.set_title("Why last month's return seems to predict", fontsize=12)


def scatter(ax):
    m = pd.DataFrame({"gold": gl, "ry": series["DFII10"].reindex(idx)}); d = pd.DataFrame({"r": 100 * np.log(m["gold"]).diff(), "x": m["ry"].diff()}).loc["2006-02":LAST].dropna()
    for sel, c, lab in ((d.index <= pd.Period("2021-12", "M"), GRAY, "2006 to 2021"), (d.index >= pd.Period("2022-01", "M"), BLUE, "2022 to 2026")):
        s = d[sel]; ax.scatter(s["x"], s["r"], s=14, color=c, alpha=0.75, label=lab); b = np.polyfit(s["x"], s["r"], 1); xs = np.array([-0.7, 0.7]); ax.plot(xs, np.polyval(b, xs), color=c if c == BLUE else INK, linewidth=2, linestyle="-" if c == BLUE else "--")
    ax.axhline(0, color=DIM, linewidth=0.8); ax.axvline(0, color=DIM, linewidth=0.8); ax.set_xlim(-0.7, 0.7); ax.set_xlabel("change in the 10-year real yield in the month (points)"); ax.set_ylabel("gold's return in the month (%)"); ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right"); ax.set_title("Gold against real yields, each month", fontsize=12)


def main():
    (HERE / "charts").mkdir(exist_ok=True)
    fig = plt.figure(figsize=(14, 8)); gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1]); a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1]); prices(a, 10.5); a.set_title("The price of gold, nominal and in today's dollars", fontsize=14)
    x = pd.PeriodIndex(RO["end_month"], freq="M").to_timestamp(); b.plot(x, 100 * RO["r2_real_yield_alone"], color=BLUE, linewidth=2.4); b.axvline(pd.Timestamp("2022-01-01"), color=ORANGE, linestyle=":", linewidth=1.3); b.set_ylabel("share of gold's monthly moves explained by real yields (%)"); b.set_ylim(0, 60); b.set_title("What real yields explain, 60 months at a time", fontsize=14)
    fig.suptitle(f"Gold took {int(LR['months below the real peak']) // 12} years to regain its 1980 real peak, and real yields now explain {100 * (1 - SI.loc['2022-01 to 2026-08', 'r2'] / SI.loc['2006-02 to 2021-12', 'r2']):.0f}% less of its moves", x=0.01, ha="left", fontsize=20, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(HERE / "charts" / "00_cover.png", dpi=100); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.3, 1]}); prices(axes[0]); decades(axes[1]); save(fig, "01_prices.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9)); drivers(axes); save(fig, "02_drivers.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8)); rolling(axes); save(fig, "03_rolling.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2)); forecast(axes); save(fig, "04_forecast.png")
    fig, ax = plt.subplots(figsize=(10, 3.8)); scatter(ax); save(fig, "05_scatter.png")


if __name__ == "__main__":
    main()
