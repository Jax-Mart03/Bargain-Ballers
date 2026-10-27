"""
Bargain Ballers - step 2: surplus-value model + visuals
========================================================
Reads clean/players.csv and clean/teams.csv (made by 01_clean.py) and writes:

  out/surplus_leaderboard.csv   every player with WAR: market value, salary, surplus
  out/no_war_salary.csv         highest-paid players with NO 2026 WAR (injured / minors / released)
  out/figs/*.png                7 charts

Usage:
  python 02_analysis_visuals.py            (finds clean/players.csv or players.csv next to the script)
  python 02_analysis_visuals.py --players "/path/to/players.csv"

Method (simple and transparent, easy to upgrade):
  market_rate  = total salary / total WAR for FA-eligible players (6+ yrs service)
  market_value = WAR * market_rate
  surplus      = market_value - salary_2026
Set MARKET_RATE below to override with your own number (or swap in quantile regression).
"""
import argparse
import importlib
import subprocess
import sys
from pathlib import Path


def _ensure(pip_name, module=None):
    try:
        importlib.import_module(module or pip_name)
    except ImportError:
        print(f"Installing missing package: {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


for _pkg in ("numpy", "pandas", "matplotlib"):
    _ensure(_pkg)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
MARKET_RATE = None          # $ per WAR; None = estimate from FA-eligible players
MIN_SALARY = 100_000        # ignore rows with ~0 salary (data artefacts) for $/WAR visuals
STAGE_COLORS = {"Pre-arb": "#2a9d8f", "Arb": "#e9c46a", "FA-eligible": "#e76f51", "Unknown": "#999999"}
plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10, "axes.titleweight": "bold",
})

try:
    from adjustText import adjust_text          # optional: pip install adjustText
except Exception:                                # pragma: no cover
    adjust_text = None


def label_points(ax, xs, ys, labels, fontsize=7.5):
    texts = [ax.text(x, y, l, fontsize=fontsize) for x, y, l in zip(xs, ys, labels)]
    if adjust_text is not None:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#888", lw=0.5))
    else:   # crude fallback: nudge labels up/right
        for t in texts:
            x, y = t.get_position()
            t.set_position((x, y))
    return texts


def fmt_m(x, _=None):
    return f"${x/1e6:,.0f}M" if abs(x) >= 1e6 else f"${x/1e3:,.0f}K"


def main(players_path: Path, out: Path):
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    print(f"Reading {players_path}")
    P = pd.read_csv(players_path)
    teams_path = players_path.parent / "teams.csv"      # optional: only needed for the team chart
    has_teams = teams_path.exists()
    T = pd.read_csv(teams_path) if has_teams else None
    if not has_teams:
        print("teams.csv not found next to players.csv -> skipping the team-efficiency chart")

    # ---------------- market rate & surplus ------------------------------
    D = P[P.war.notna() & (P.salary_2026 >= MIN_SALARY)].copy()
    fa = D[D.stage == "FA-eligible"]
    rate = MARKET_RATE or fa.salary_2026.sum() / fa.war.sum()
    print(f"Market rate: ${rate/1e6:.2f}M per WAR "
          f"({'override' if MARKET_RATE else f'FA-eligible, n={len(fa)}'})")
    D["market_value"] = D.war * rate
    D["surplus"] = D.market_value - D.salary_2026
    D["dollars_per_war"] = np.where(D.war > 0, D.salary_2026 / D.war, np.nan)
    cols = ["player", "team", "pos", "pos_group", "age", "mls", "stage", "years_remaining",
            "contract_text", "salary_2026", "war", "pa", "ip", "market_value", "surplus", "dollars_per_war",
            "low_sample", "name_only_match", "war_name_collision"]
    D.sort_values("surplus", ascending=False)[cols].to_csv(out / "surplus_leaderboard.csv", index=False)

    nowar = (P[P.war_missing & (P.salary_2026 > 0)]
             .sort_values("salary_2026", ascending=False)
             [["player", "team", "pos", "age", "stage", "contract_text", "salary_2026"]].head(50))
    nowar.to_csv(out / "no_war_salary.csv", index=False)
    print(f"Salary paid to {int(P.war_missing.sum())} players with no 2026 WAR: "
          f"${P.loc[P.war_missing,'salary_2026'].sum()/1e6:,.0f}M "
          f"(top: {', '.join(nowar.player.head(3))})")

    top = D.nlargest(12, "surplus")
    bot = D.nsmallest(12, "surplus")

    # ---------------- 1. salary vs WAR scatter ---------------------------
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for st, g in D.groupby("stage"):
        ax.scatter(g.salary_2026, g.war, s=14, alpha=.6, c=STAGE_COLORS.get(st), label=st, lw=0)
    xs = np.logspace(np.log10(MIN_SALARY), np.log10(D.salary_2026.max()), 50)
    ax.plot(xs, xs / rate, "k--", lw=1, label=f"Market line (${rate/1e6:.1f}M/WAR)")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_m))
    ax.set_xlabel("2026 salary (log scale)")
    ax.set_ylabel("2026 WAR")
    ax.set_title("Who beats the market? Salary vs WAR (above the dashed line = bargain)")
    sel = pd.concat([D.nlargest(7, "war"), top.head(6), bot.head(7)]).drop_duplicates("player")
    label_points(ax, sel.salary_2026, sel.war, sel.player)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(figs / "1_salary_vs_war.png"); plt.close(fig)

    # ---------------- 2. $/WAR by service stage --------------------------
    g = (D[D.stage != "Unknown"].groupby("stage")
         .agg(salary=("salary_2026", "sum"), war=("war", "sum"), n=("player", "size")))
    g["dpw"] = g.salary / g.war
    g = g.loc[["Pre-arb", "Arb", "FA-eligible"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(g.index, g.dpw / 1e6, color=[STAGE_COLORS[s] for s in g.index])
    for b, (_, r) in zip(bars, g.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"${r.dpw/1e6:.1f}M\n(n={int(r.n)})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("$ millions per WAR"); ax.set_ylim(0, g.dpw.max() / 1e6 * 1.25)
    ax.set_title("The service-time arbitrage: cost of one win")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(figs / "2_dollars_per_war_by_stage.png"); plt.close(fig)

    # ---------------- 3. team efficiency ---------------------------------
    t = None
    if has_teams:
        t = T.dropna(subset=["roster_war", "payroll_all_rows"]).copy()
        slope, intercept = np.polyfit(t.payroll_all_rows / 1e6, t.roster_war, 1)
        t["expected_war"] = intercept + slope * t.payroll_all_rows / 1e6
        t["war_over_expected"] = t.roster_war - t.expected_war
        t.sort_values("war_over_expected", ascending=False).to_csv(out / "team_efficiency.csv", index=False)
        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        sc = ax.scatter(t.payroll_all_rows / 1e6, t.roster_war, c=t.war_over_expected, cmap="RdYlGn",
                        s=70, edgecolor="k", lw=.4)
        xs = np.linspace(t.payroll_all_rows.min() / 1e6, t.payroll_all_rows.max() / 1e6, 20)
        ax.plot(xs, intercept + slope * xs, "k--", lw=1)
        label_points(ax, t.payroll_all_rows / 1e6, t.roster_war, t.team, fontsize=8.5)
        ax.set_xlabel("2026 payroll incl. dead money ($M)"); ax.set_ylabel("Roster WAR (players on current team)")
        ax.set_title("Team efficiency: payroll vs WAR (green = more WAR than payroll predicts)")
        plt.colorbar(sc, label="WAR over expected")
        fig.tight_layout(); fig.savefig(figs / "3_team_efficiency.png"); plt.close(fig)

    # ---------------- 4. surplus leaders / laggards ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, d, color, ttl in [(axes[0], top.iloc[::-1], "#2a9d8f", "Biggest bargains (surplus $M)"),
                              (axes[1], bot.iloc[::-1], "#e76f51", "Biggest overpays (surplus $M)")]:
        ax.barh(d.player + " (" + d.team + ")", d.surplus / 1e6, color=color)
        ax.set_title(ttl); ax.grid(axis="y", visible=False)
        ax.set_xlabel("WAR x market rate - salary")
    fig.tight_layout(); fig.savefig(figs / "4_surplus_leaders.png"); plt.close(fig)

    # ---------------- 5. control window ----------------------------------
    cw = D[D.mls.notna() & (D.mls < 7)]
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(cw.mls, cw.surplus / 1e6, c=cw.war, cmap="viridis", s=18, alpha=.8)
    ax.axvline(3, color="gray", ls=":"); ax.axvline(6, color="gray", ls=":")
    ax.text(1.4, ax.get_ylim()[1] * .92, "Pre-arb", ha="center"); ax.text(4.5, ax.get_ylim()[1] * .92, "Arb", ha="center")
    ax.text(6.5, ax.get_ylim()[1] * .92, "FA", ha="center")
    pick = cw[cw.mls >= 2.5].nlargest(10, "surplus")
    label_points(ax, pick.mls, pick.surplus / 1e6, pick.player)
    ax.set_xlabel("Years of MLB service (Jan 2026)"); ax.set_ylabel("2026 surplus ($M)")
    ax.set_title("Control window: high-surplus players whose cheap years are running out")
    plt.colorbar(sc, label="2026 WAR")
    fig.tight_layout(); fig.savefig(figs / "5_control_window.png"); plt.close(fig)

    # ---------------- 6. position market ---------------------------------
    pg = (D[D.pos_group != "UNK"].groupby("pos_group")
          .agg(salary=("salary_2026", "sum"), war=("war", "sum"), n=("player", "size")))
    pg = pg[pg.war > 0]; pg["dpw"] = pg.salary / pg.war
    pg = pg.sort_values("dpw")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(pg.index, pg.dpw / 1e6, color="#264653")
    for i, (idx, r) in enumerate(pg.iterrows()):
        ax.text(r.dpw / 1e6, i, f"  ${r.dpw/1e6:.1f}M (n={int(r.n)})", va="center", fontsize=8.5)
    ax.set_xlabel("$ millions per WAR (all players)"); ax.grid(axis="y", visible=False)
    ax.set_title("Cost of a win by position group")
    ax.set_xlim(0, pg.dpw.max() / 1e6 * 1.3)
    fig.tight_layout(); fig.savefig(figs / "6_position_market.png"); plt.close(fig)

    # ---------------- 7. age curve ---------------------------------------
    ag = D[D.age.between(20, 42)].copy()
    ag["age_bin"] = ag.age.astype(int)
    a = ag.groupby("age_bin").agg(war=("war", "mean"), sal=("salary_2026", "mean"), n=("war", "size"))
    a = a[a.n >= 8]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(a.index, a.war, "o-", color="#2a9d8f", label="Mean WAR")
    ax.set_xlabel("Age (7/1/26)"); ax.set_ylabel("Mean WAR", color="#2a9d8f")
    ax2 = ax.twinx(); ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax2.plot(a.index, a.sal / 1e6, "s--", color="#e76f51", label="Mean salary")
    ax2.set_ylabel("Mean salary ($M)", color="#e76f51")
    ax.set_title("Age curve: teams pay for the past (salary peaks after production)")
    fig.tight_layout(); fig.savefig(figs / "7_age_curve.png"); plt.close(fig)

    print("\nTop 10 bargains:\n", top.head(10)[["player", "team", "salary_2026", "war", "surplus"]].to_string(index=False))
    print("\nTop 10 overpays:\n", bot.head(10)[["player", "team", "salary_2026", "war", "surplus"]].to_string(index=False))
    if t is not None:
        print("\nTeams WAR over expected:\n",
              t.sort_values("war_over_expected")[["team", "payroll_all_rows", "roster_war", "war_over_expected"]]
              .iloc[[0, 1, 2, -3, -2, -1]].to_string(index=False))
    print(f"\nWrote outputs to {out}/")


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent          # folder this script lives in
    default_players = next((p for p in [HERE / "clean" / "players.csv", HERE / "players.csv"] if p.exists()),
                           HERE / "clean" / "players.csv")
    ap = argparse.ArgumentParser(description="Surplus model + charts from players.csv")
    ap.add_argument("--players", default=str(default_players), help="path to players.csv")
    ap.add_argument("--out", default=str(HERE / "out"), help="output folder")
    a, _stray = ap.parse_known_args()                 # stray/extra text is ignored
    pp = Path(a.players)
    if not pp.exists():
        raise SystemExit(f"\nCould not find players.csv at:\n  {pp}\nRun 01_clean.py first (it creates clean/players.csv).")
    main(pp, Path(a.out))
