"""
Bargain Ballers - step 1: cleaning pipeline
============================================
Reads the `League` sheet (the division sheets are duplicates of it) and writes
tidy, analysis-ready tables:

  clean/players.csv      one row per player (contract + 2026 WAR + flags)
  clean/adjustments.csv  non-contract money rows (releases, buyouts, bonuses, trade credits...)
  clean/teams.csv        one row per team (payroll, dead money, WAR, efficiency)
  clean/qa_report.txt    what was found / fixed / flagged

Usage:
  python 01_clean.py --input Bargain_Ballers_Updated.xlsx --outdir clean

Key modelling decisions (all easy to change in the CONFIG block):
  * A row is a "contract row" if it has a position; otherwise it is an "adjustment row".
  * Traded players show up on several rows. Salary is SUMMED per player so the
    total reflects what the player cost in 2026 (this reproduces the stated
    contract value for 1-year deals, e.g. Civale = $6.0M).
  * WAR that is missing stays NaN. NaN != 0 (per the workbook's Match Guide).
  * Impossible ages (<18 or >45) are set to NaN and flagged, not dropped.
"""
import argparse
import importlib
import re
import subprocess
import sys
from pathlib import Path


def _ensure(pip_name, module=None):
    """Import a package, installing it with pip first if it's missing."""
    try:
        importlib.import_module(module or pip_name)
    except ImportError:
        print(f"Installing missing package: {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


for _pkg in ("numpy", "pandas", "openpyxl"):
    _ensure(_pkg)

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
SEASON = 2026
AGE_MIN, AGE_MAX = 18, 45          # outside this range -> age set to NaN + flagged
PRE_ARB_MAX_MLS = 3.0              # < 3 years service  -> pre-arb (proxy; ignores Super Two)
ARB_MAX_MLS = 6.0                  # 3 to < 6           -> arbitration; 6+ -> free agent
LOW_PA, LOW_IP = 100, 20           # small-sample flags
# Adjustment categories whose money belongs to the *player's 2026 cost* and get attached to him.
ATTACH_CATEGORIES = {"trade", "claimed", "dfa", "bonus", "cbt_credit", "other"}
# Categories that represent money paid to players who are no longer producing for that team.
DEAD_MONEY_CATEGORIES = {"released", "buyout", "outrighted"}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def display_name(raw: str) -> str:
    """'Judge, Aaron*' -> 'Aaron Judge'"""
    raw = str(raw).replace("*", "").strip()
    if "," in raw:
        last, first = [s.strip() for s in raw.split(",", 1)]
        return f"{first} {last}".strip()
    return raw


def pos_group(pos: str) -> str:
    if not isinstance(pos, str):
        return "UNK"
    p = pos.lower()
    if "dh-rhp" in p or "dh-lhp" in p or "rhp-dh" in p or "lhp-dh" in p:
        return "TWP"                                   # two-way
    if p.endswith("-s") and ("rhp" in p or "lhp" in p):
        return "SP"
    if p.endswith("-c") and ("rhp" in p or "lhp" in p):
        return "CL"
    if p in ("rhp", "lhp"):
        return "RP"                                    # assumption: bare rhp/lhp = reliever
    first = p.split("-")[0]
    return {"c": "C", "1b": "1B", "2b": "2B", "3b": "3B", "ss": "SS",
            "lf": "OF", "cf": "OF", "rf": "OF", "of": "OF", "dh": "DH"}.get(first, "UNK")


def _money(num: str, unit: str | None) -> float:
    v = float(num.replace(",", ""))
    unit = (unit or "").upper()
    if unit == "M":
        return v * 1e6
    if unit == "K":
        return v * 1e3
    if unit == "B":
        return v * 1e9
    return v * 1e6 if v < 1000 else v    # '$24' with no suffix -> assume millions; '$784,550' -> dollars


CONTRACT_RE = re.compile(
    r"(?P<yrs>\d+)\s*y(?:r|ears)?\b\s*(?:/\s*\$+\s*(?P<num>[\d,\.]+)\s*(?P<unit>[MmKkBb])?)?"
    r"\s*\((?P<start>\d{2})(?:\s*-\s*(?P<end>\d{2}))?\)"
)


def parse_contract(text) -> dict:
    out = dict(contract_years=np.nan, contract_total=np.nan, contract_aav=np.nan,
               contract_start=np.nan, contract_end=np.nan, years_remaining=np.nan,
               has_option=False, multi_contract=False, contract_parse_ok=False)
    if not isinstance(text, str):
        return out
    matches = list(CONTRACT_RE.finditer(text))
    if not matches:
        return out
    m = matches[0]                                    # first contract = current deal
    yrs = int(m["yrs"])
    start = 2000 + int(m["start"])
    end = 2000 + int(m["end"]) if m["end"] else start
    total = _money(m["num"], m["unit"]) if m["num"] else np.nan
    out.update(
        contract_years=yrs, contract_total=total,
        contract_aav=total / yrs if (yrs and not np.isnan(total)) else np.nan,
        contract_start=start, contract_end=end,
        years_remaining=max(end - SEASON, 0),         # seasons AFTER the current one
        has_option=bool(re.search(r"opt", text, re.I)),
        multi_contract=len(matches) > 1, contract_parse_ok=True,
    )
    return out


def categorize_adjustment(text) -> str:
    t = str(text).lower()
    if "cbt credit" in t:
        return "cbt_credit"
    if "buyout" in t:
        return "buyout"
    if "outrighted" in t:
        return "outrighted"
    if "released" in t:
        return "released"
    if t.startswith("dfa"):
        return "dfa"
    if "claimed" in t:
        return "claimed"
    if "bonus" in t:
        return "bonus"
    if "trade" in t or t.startswith("traded") or t.startswith("from ") or t.startswith("to "):
        return "trade"
    return "other"


def parse_round(rd):
    """'1s' -> (1, True); 7 -> (7, False); NaN -> (nan, False)"""
    if pd.isna(rd):
        return np.nan, False
    s = str(rd).strip().lower()
    m = re.match(r"(\d+)(s?)", s)
    if not m:
        return np.nan, False
    return float(m.group(1)), m.group(2) == "s"


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(inp: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    qa = []

    def log(msg=""):
        qa.append(str(msg))
        print(msg)

    sheets = pd.ExcelFile(inp).sheet_names
    for needed in ("League", "WAR Leaderboard"):
        if needed not in sheets:
            raise SystemExit(f"\nSheet '{needed}' not found. Sheets in your workbook: {sheets}")
    raw = pd.read_excel(inp, sheet_name="League")
    war_board = pd.read_excel(inp, sheet_name="WAR Leaderboard")
    log(f"League rows read: {len(raw):,}")

    df = raw.rename(columns={
        "TEAM": "team", "Player": "player_raw", "Pos.": "pos", "Year": "draft_year",
        "Rd": "rd_raw", "Pick": "pick_raw", "AGE (7/1/26)": "age_raw", "MLS (1/26)": "mls",
        "Opts left 1/26": "options_raw", "Agent": "agent", "Length / Total Value": "contract_text",
        "LRD": "salary", "Competitive Balance": "cbt", "Player Match Key": "match_key",
        "WAR Match Status": "war_match_status", "WAR Rank": "war_rank", "WAR PA": "pa",
        "WAR IP": "ip", "Bat WAR": "bat_war", "Pit WAR": "pit_war", "WAR": "war",
    }).drop(columns=[c for c in raw.columns if str(c).startswith("Unnamed")], errors="ignore")
    df["name_norm"] = df["match_key"].str.split("|").str[0]
    df["is_contract_row"] = df["pos"].notna()
    log(f"Contract rows: {df.is_contract_row.sum():,} | adjustment rows: {(~df.is_contract_row).sum():,}")

    # ---------------- contract rows -> player entities --------------------
    C = df[df.is_contract_row].copy()
    C["age"] = C["age_raw"]
    bad_age = C["age"].lt(AGE_MIN) | C["age"].gt(AGE_MAX)
    C["age_flag"] = np.where(bad_age, "suspect_age_set_nan", "")
    C.loc[bad_age, "age"] = np.nan
    log(f"Suspect ages set to NaN: {bad_age.sum()} "
        f"({', '.join(C.loc[bad_age,'player_raw'] + ' [' + C.loc[bad_age,'age_raw'].astype(int).astype(str) + ']')})")

    # entity id = normalised name + raw age (separates the two Max Muncys, Fermins, Martes...)
    C["player_id"] = C["name_norm"] + "|" + C["age_raw"].fillna(-1).astype(int).astype(str)

    # a traded player can have a contract row on both teams -> keep one row (prefer exact team match)
    dup_mask = C.duplicated("player_id", keep=False)
    if dup_mask.any():
        log(f"Players with contract rows on >1 team (traded): "
            f"{C.loc[dup_mask,'player_raw'].nunique()} -> {sorted(C.loc[dup_mask,'player_raw'].unique())}")
    C["_pref"] = (C["war_match_status"].astype(str).str.contains("team")).astype(int)
    C = (C.sort_values(["player_id", "_pref"], ascending=[True, False])
           .drop_duplicates("player_id", keep="first").drop(columns="_pref"))

    # ---------------- adjustment rows ------------------------------------
    A = df[~df.is_contract_row].copy()
    A["category"] = A["contract_text"].map(categorize_adjustment)
    A["note"] = A["contract_text"]

    # attach to a player: by unique name, or by (name, team) if the name is ambiguous
    name_counts = C.groupby("name_norm")["player_id"].nunique()
    team_lookup = C.set_index(["name_norm", "team"])["player_id"].to_dict()
    uniq_lookup = C[C["name_norm"].map(name_counts) == 1].set_index("name_norm")["player_id"].to_dict()

    def assign(row):
        if row["name_norm"] in uniq_lookup:
            return uniq_lookup[row["name_norm"]]
        return team_lookup.get((row["name_norm"], row["team"]), np.nan)

    A["player_id"] = A.apply(assign, axis=1)
    A["attached"] = A["player_id"].notna() & A["category"].isin(ATTACH_CATEGORIES)
    log("Adjustment categories:\n" + A["category"].value_counts().to_string())
    log(f"Adjustment rows attached to a player's 2026 cost: {A.attached.sum():,}")

    # total salary / CBT per player = contract row + attached adjustments (incl. negative trade offsets)
    money_rows = pd.concat([C[["player_id", "salary", "cbt"]],
                            A.loc[A.attached, ["player_id", "salary", "cbt"]]])
    tot = money_rows.groupby("player_id").agg(
        salary_2026=("salary", lambda s: s.sum(min_count=1)),
        cbt_2026=("cbt", lambda s: s.sum(min_count=1)))

    # ---------------- build players table --------------------------------
    P = C.merge(tot, on="player_id", how="left")
    P["player"] = P["player_raw"].map(display_name)
    P["starred"] = P["player_raw"].str.contains(r"\*", regex=True)
    P["pos_group"] = P["pos"].map(pos_group)
    P["role"] = np.where(P.pos_group == "TWP", "Two-way",
                 np.where(P.pos_group.isin(["SP", "RP", "CL"]), "Pitcher", "Hitter"))

    def stage(m):
        if pd.isna(m):
            return "Unknown"
        return "Pre-arb" if m < PRE_ARB_MAX_MLS else ("Arb" if m < ARB_MAX_MLS else "FA-eligible")
    P["stage"] = P["mls"].map(stage)

    opt = P["options_raw"].astype(str).str.extract(r"(\d)\s*/\s*(\d)")
    P["options_left"], P["options_total"] = pd.to_numeric(opt[0]), pd.to_numeric(opt[1])

    rd = P["rd_raw"].map(parse_round)
    P["draft_round"] = [r[0] for r in rd]
    P["supplemental_round"] = [r[1] for r in rd]
    pick_num = pd.to_numeric(P["pick_raw"], errors="coerce")
    P["draft_pick"] = pick_num
    P["origin"] = np.where(pick_num.notna(), "Draft",
                    P["pick_raw"].astype(str).where(P["pick_raw"].notna(), "Unknown"))
    P["is_international"] = ~P["origin"].isin(["Draft", "Unknown", "USA"])

    contract = pd.DataFrame([parse_contract(t) for t in P["contract_text"]], index=P.index)
    P = pd.concat([P, contract], axis=1)
    log(f"Contracts parsed: {P.contract_parse_ok.sum():,}/{len(P):,} "
        f"| multi-contract strings: {P.multi_contract.sum()} | with option language: {P.has_option.sum()}")
    unparsed = P.loc[~P.contract_parse_ok, "contract_text"].dropna().unique()
    log(f"Unparsed contract strings (sample): {list(unparsed[:8])}")

    # ---------------- WAR features ---------------------------------------
    P["war_missing"] = P["war"].isna()
    P["war_per_600pa"] = np.where(P["pa"] >= LOW_PA, P["war"] / P["pa"] * 600, np.nan)
    P["war_per_200ip"] = np.where(P["ip"] >= LOW_IP, P["war"] / P["ip"] * 200, np.nan)
    P["low_sample"] = ((P["pa"].fillna(0) < LOW_PA) & (P["ip"].fillna(0) < LOW_IP)) & ~P["war_missing"]
    P["name_only_match"] = P["war_match_status"].eq("Matched (name)")

    # name collisions on the WAR board (same normalised name appears >1 time)
    wb_names = war_board["Player Match Key"].str.split("|").str[0]
    collide = set(wb_names[wb_names.duplicated(keep=False)])
    P["war_name_collision"] = P["name_norm"].isin(collide)
    log(f"Names duplicated on WAR leaderboard (possible mismatches): {sorted(collide)}")
    log(f"Name-only WAR matches among players: {P.name_only_match.sum()} "
        f"(mostly traded players; spot-check those flagged war_name_collision)")

    # ---------------- QA: salary sums vs stated 1-year totals ------------
    one_yr = P[(P.contract_years == 1) & P.contract_total.notna() & P.salary_2026.notna()]
    if len(one_yr):
        diff = (one_yr["salary_2026"] - one_yr["contract_total"]).abs() / one_yr["contract_total"]
        log(f"Salary sum vs stated total on {len(one_yr)} 1-yr deals: "
            f"{(diff < 0.02).mean():.0%} within 2% (median abs diff {diff.median():.1%}). "
            f"Most misses are LOWER than stated = prorated pay (minor-league time, mid-season release); "
            f"salary_2026 is treated as actual 2026 cost.")
    neg = P[P.salary_2026 < 0]
    log(f"Players with negative total salary after offsets: {len(neg)}")
    log(f"Players with no salary at all: {P.salary_2026.isna().sum()}")
    log(f"Players with no 2026 WAR (NaN, not zero): {P.war_missing.sum()}")

    keep = ["player_id", "player", "team", "pos", "pos_group", "role", "age", "age_flag", "mls", "stage",
            "options_left", "options_total", "agent", "draft_year", "draft_round", "supplemental_round",
            "draft_pick", "origin", "is_international", "contract_text", "contract_years", "contract_total",
            "contract_aav", "contract_start", "contract_end", "years_remaining", "has_option",
            "multi_contract", "starred", "salary_2026", "cbt_2026", "war", "bat_war", "pit_war", "pa", "ip",
            "war_rank", "war_missing", "war_per_600pa", "war_per_200ip", "low_sample",
            "war_match_status", "name_only_match", "war_name_collision", "match_key"]
    P = P[keep].sort_values(["team", "salary_2026"], ascending=[True, False])

    # ---------------- team table -----------------------------------------
    payroll = df.groupby("team")["salary"].sum().rename("payroll_all_rows")
    dead = (A[A.category.isin(DEAD_MONEY_CATEGORIES)].groupby("team")["salary"].sum()
            .clip(lower=0).rename("dead_money"))
    pt = P.groupby("team").agg(
        n_players=("player_id", "size"),
        roster_war=("war", "sum"),
        pre_arb_war=("war", lambda s: s[P.loc[s.index, "stage"] == "Pre-arb"].sum()),
        top5_salary=("salary_2026", lambda s: s.nlargest(5).sum()),
        player_salary=("salary_2026", "sum"))
    T = pd.concat([payroll, dead, pt], axis=1).fillna({"dead_money": 0})
    T["dead_money_share"] = T["dead_money"] / T["payroll_all_rows"]
    T["top5_share"] = T["top5_salary"] / T["player_salary"]
    T["dollars_per_war"] = T["payroll_all_rows"] / T["roster_war"]
    T = T.reset_index().sort_values("dollars_per_war")
    log(f"Teams: {len(T)} | total payroll in data: ${T.payroll_all_rows.sum()/1e9:.2f}B "
        f"| dead money: ${T.dead_money.sum()/1e6:.0f}M")
    log("NOTE: team WAR credits a traded player's full WAR to his current team only.")

    P.to_csv(outdir / "players.csv", index=False)
    A[["team", "player_raw", "name_norm", "category", "note", "salary", "cbt", "player_id", "attached"]] \
        .to_csv(outdir / "adjustments.csv", index=False)
    T.to_csv(outdir / "teams.csv", index=False)
    (outdir / "qa_report.txt").write_text("\n".join(qa))
    log(f"\nWrote players.csv ({len(P):,}), adjustments.csv ({len(A):,}), teams.csv ({len(T)}) to {outdir}/")


def find_workbook(folder: Path) -> Path:
    """Find the Bargain Ballers workbook next to this script, whatever it's called."""
    files = [p for p in folder.glob("*.xlsx") if not p.name.startswith("~$")]
    pref = [p for p in files if "bargain" in p.name.lower()]
    if pref:
        return sorted(pref)[0]
    if files:
        return sorted(files)[0]
    raise SystemExit(f"\nNo .xlsx workbook found in:\n  {folder}\n"
                     f"Put your workbook in that folder (next to this script) and run again.")


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent            # folder this script lives in
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="optional: path to the workbook")
    ap.add_argument("--outdir", default=None, help="optional: output folder")
    a, _stray = ap.parse_known_args()                 # stray/extra text is ignored, not an error
    inp = Path(a.input) if a.input else find_workbook(HERE)
    if not inp.exists() and (HERE / inp).exists():    # relative name -> look next to the script
        inp = HERE / inp
    outdir = Path(a.outdir) if a.outdir else HERE / "clean"
    if not outdir.is_absolute():
        outdir = HERE / outdir
    print(f"Workbook: {inp}\nOutput:   {outdir}\n")
    main(inp, outdir)
