import io
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# Transition matrices + greedy allocator (integrated)
# ============================================================

# --- Segment B: donors (B1) -> receivers (B2) ---
B1_DONORS = ["B1_SME_TERM", "B1_MIDCORP_NONIG", "B1_EM_CORP", "B1_CRE_NON_HVCRE"]
B2_RECEIVERS = ["B2_PRIME_MORTGAGES", "B2_TRANSACTION_BANKING", "B2_ASSET_BACKED", "B2_PRIME_CONSUMER"]

# ΔRWA density (pp) for B (negative = reduces RWAs per unit of exposure)
DELTA_RWA_PP_B: Dict[Tuple[str, str], float] = {
    ("B1_SME_TERM",      "B2_PRIME_MORTGAGES"):     -3.000,
    ("B1_SME_TERM",      "B2_TRANSACTION_BANKING"): -3.600,
    ("B1_SME_TERM",      "B2_ASSET_BACKED"):        -2.000,
    ("B1_SME_TERM",      "B2_PRIME_CONSUMER"):      -1.636,

    ("B1_MIDCORP_NONIG", "B2_PRIME_MORTGAGES"):     -2.667,
    ("B1_MIDCORP_NONIG", "B2_TRANSACTION_BANKING"): -3.200,
    ("B1_MIDCORP_NONIG", "B2_ASSET_BACKED"):        -1.778,
    ("B1_MIDCORP_NONIG", "B2_PRIME_CONSUMER"):      -1.455,

    ("B1_EM_CORP",       "B2_PRIME_MORTGAGES"):     -3.333,
    ("B1_EM_CORP",       "B2_TRANSACTION_BANKING"): -4.000,
    ("B1_EM_CORP",       "B2_ASSET_BACKED"):        -2.222,
    ("B1_EM_CORP",       "B2_PRIME_CONSUMER"):      -1.818,

    ("B1_CRE_NON_HVCRE", "B2_PRIME_MORTGAGES"):     -3.167,
    ("B1_CRE_NON_HVCRE", "B2_TRANSACTION_BANKING"): -3.800,
    ("B1_CRE_NON_HVCRE", "B2_ASSET_BACKED"):        -2.111,
    ("B1_CRE_NON_HVCRE", "B2_PRIME_CONSUMER"):      -1.727,
}


# ΔNet spread (bps) for B (positive = increases net margin)
DELTA_SPREAD_BPS_B: Dict[Tuple[str, str], float] = {
    ("B1_SME_TERM",       "B2_PRIME_MORTGAGES"):      -80,
    ("B1_SME_TERM",       "B2_TRANSACTION_BANKING"):  -30,
    ("B1_SME_TERM",       "B2_ASSET_BACKED"):         +100,
    ("B1_SME_TERM",       "B2_PRIME_CONSUMER"):       +240,

    ("B1_MIDCORP_NONIG",  "B2_PRIME_MORTGAGES"):      -40,
    ("B1_MIDCORP_NONIG",  "B2_TRANSACTION_BANKING"):  +10,
    ("B1_MIDCORP_NONIG",  "B2_ASSET_BACKED"):         +140,
    ("B1_MIDCORP_NONIG",  "B2_PRIME_CONSUMER"):       +280,

    ("B1_EM_CORP",        "B2_PRIME_MORTGAGES"):      -120,
    ("B1_EM_CORP",        "B2_TRANSACTION_BANKING"):  -70,
    ("B1_EM_CORP",        "B2_ASSET_BACKED"):         +60,
    ("B1_EM_CORP",        "B2_PRIME_CONSUMER"):       +200,

    ("B1_CRE_NON_HVCRE",  "B2_PRIME_MORTGAGES"):      -100,
    ("B1_CRE_NON_HVCRE",  "B2_TRANSACTION_BANKING"):  -50,
    ("B1_CRE_NON_HVCRE",  "B2_ASSET_BACKED"):         +80,
    ("B1_CRE_NON_HVCRE",  "B2_PRIME_CONSUMER"):       +220,
}


# Absolute net spread (bps) by receiver (same for all donors)
ABS_NET_SPREAD_BPS_BY_B2: Dict[str, float] = {
    "B2_PRIME_MORTGAGES": 100,
    "B2_TRANSACTION_BANKING": 150,
    "B2_ASSET_BACKED": 280,
    "B2_PRIME_CONSUMER": 420,
}



@dataclass(frozen=True)
class Cell:
    donor: str
    receiver: str
    delta_rwa_pp: float  # ΔRWA density (pp), negative means RWAs fall per unit exposure moved
    donor_risk_weight: float  # decimal, e.g. 0.90
    receiver_risk_weight: float  # decimal, e.g. 0.30
    abs_net_spread_bps: float  # receiver absolute net spread (bps)
    delta_s_eff_dec: float  # effective spread-like term (decimal), per new formula
    ratio: float  # delta_s_eff_dec per unit of RWA reduction (profitability per RWA)

@dataclass
class Allocation:
    donor: str
    receiver: str
    exposure_used_eur_bn: float
    rwa_reduction_eur_bn: float
    assets_redeploy_used_eur_bn: float
    donor_risk_weight: float
    receiver_risk_weight: float
    delta_rwa_pp: float
    abs_net_spread_bps: float
    delta_s_eff_dec: float
    ratio: float



# Risk weights used for the transition engine
DONOR_RISK_WEIGHT: Dict[str, float] = {
    "B1_SME_TERM": 0.90,
    "B1_MIDCORP_NONIG": 0.80,
    "B1_EM_CORP": 1.00,
    "B1_CRE_NON_HVCRE": 0.95,
}

RECEIVER_RISK_WEIGHT: Dict[str, float] = {
    "B2_PRIME_MORTGAGES": 0.30,
    "B2_TRANSACTION_BANKING": 0.25,
    "B2_ASSET_BACKED": 0.45,
    "B2_PRIME_CONSUMER": 0.55,
}


def _best_cell_per_donor(
    donors: List[str],
    srt_efficiency: float,
    srt_cost_dec: float,
    delta_rwa_pp: Dict[Tuple[str, str], float],
) -> Dict[str, Cell]:
    """Pick the best receiver per donor using the new delta_s_eff formula.

    Eligibility:
      - delta_rwa_pp must be negative (RWA density improvement)
      - delta_s_eff_bps must be > 1.0
    Ranking metric:
      - ratio = delta_s_eff_dec / (donor_risk_weight * srt_efficiency)
        i.e., profitability per unit of RWA reduction.
    """
    best: Dict[str, Cell] = {}
    eff = float(srt_efficiency)

    for (d, r), drwa in delta_rwa_pp.items():
        if d not in donors:
            continue

        donor_rw = float(DONOR_RISK_WEIGHT.get(d, 0.0))
        receiver_rw = float(RECEIVER_RISK_WEIGHT.get(r, 0.0))
        abs_spread_bps = float(ABS_NET_SPREAD_BPS_BY_B2.get(r, 0.0))

        if donor_rw <= 0 or receiver_rw <= 0:
            continue

        # must improve RWA density
        if not (drwa < 0):
            continue

        delta_rwa_density = (-float(drwa))  # use as-is (quotient/multiplier, positive)
        abs_spread_bps = abs_spread_bps  # already in bps

        # New economics term (work in bps, then convert to decimal)
        srt_cost_bp = float(srt_cost_dec) * 10000.0
        delta_s_eff_bps = delta_rwa_density * eff * abs_spread_bps - 2.0 * srt_cost_bp
        delta_s_eff_dec = delta_s_eff_bps / 10000.0

        if delta_s_eff_bps <= 1.0:
            continue

        rwa_red_per_eur = donor_rw * eff
        if rwa_red_per_eur <= 0:
            continue

        ratio = delta_s_eff_dec / rwa_red_per_eur

        cand = Cell(
            donor=d,
            receiver=r,
            delta_rwa_pp=float(drwa),
            donor_risk_weight=donor_rw,
            receiver_risk_weight=receiver_rw,
            abs_net_spread_bps=abs_spread_bps,
            delta_s_eff_dec=float(delta_s_eff_dec),
            ratio=float(ratio),
        )

        if (d not in best) or (cand.ratio > best[d].ratio):
            best[d] = cand

    return best


def greedy_allocate_rwa_reduction(
    rwa_target_eur_bn: float,
    donor_exposure_eur_bn: Dict[str, float],
    srt_efficiency: float,
    srt_cost_dec: float,
) -> Dict[str, object]:
    """Greedy allocator (Segment B only).

    RWA reduction achieved per allocation is computed as:
        rwa_reduction = exposure_used * donor_risk_weight * srt_efficiency

    Assets redeployed used per allocation is computed as:
        assets_redeploy = rwa_reduction / receiver_risk_weight
    """
    donors = [d for d in B1_DONORS if donor_exposure_eur_bn.get(d, 0.0) > 0]

    best = _best_cell_per_donor(
        donors=donors,
        srt_efficiency=float(srt_efficiency),
        srt_cost_dec=float(srt_cost_dec),
        delta_rwa_pp=DELTA_RWA_PP_B,
    )

    if not best:
        return {
            "allocations": [],
            "total_rwa_reduction_eur_bn": 0.0,
            "total_exposure_used_eur_bn": 0.0,
            "total_assets_redeploy_used_eur_bn": 0.0,
            "remaining_rwa_target_eur_bn": float(rwa_target_eur_bn),
            "ranked_donors": [],
            "status": "NO_ELIGIBLE_TRANSITIONS",
        }

    ranked = sorted(best.keys(), key=lambda d: best[d].ratio, reverse=True)

    remaining = float(rwa_target_eur_bn)
    allocs: List[Allocation] = []
    total_rwa_red = 0.0
    total_expo = 0.0
    total_assets_redeploy = 0.0

    eff = float(srt_efficiency)

    for d in ranked:
        if remaining <= 0:
            break

        cell = best[d]
        expo_avail = float(donor_exposure_eur_bn.get(d, 0.0))
        if expo_avail <= 0:
            continue

        # NEW: RWA reduction per EUR exposure moved
        red_per_eur = float(cell.donor_risk_weight) * eff
        if red_per_eur <= 0:
            continue

        expo_needed = remaining / red_per_eur
        expo_used = min(expo_avail, expo_needed)

        rwa_red = expo_used * red_per_eur

        # NEW: assets redeployed used on the receiver side
        assets_redeploy = rwa_red / float(cell.receiver_risk_weight)

        allocs.append(
            Allocation(
                donor=cell.donor,
                receiver=cell.receiver,
                exposure_used_eur_bn=expo_used,
                rwa_reduction_eur_bn=rwa_red,
                assets_redeploy_used_eur_bn=assets_redeploy,
                donor_risk_weight=cell.donor_risk_weight,
                receiver_risk_weight=cell.receiver_risk_weight,
                delta_rwa_pp=cell.delta_rwa_pp,
                abs_net_spread_bps=cell.abs_net_spread_bps,
                delta_s_eff_dec=cell.delta_s_eff_dec,
                ratio=cell.ratio,
            )
        )

        total_expo += expo_used
        total_rwa_red += rwa_red
        total_assets_redeploy += assets_redeploy
        remaining -= rwa_red

    status = "OK" if remaining <= 1e-9 else "TARGET_NOT_MET"
    return {
        "allocations": allocs,
        "total_rwa_reduction_eur_bn": total_rwa_red,
        "total_exposure_used_eur_bn": total_expo,
        "total_assets_redeploy_used_eur_bn": total_assets_redeploy,
        "remaining_rwa_target_eur_bn": max(0.0, remaining),
        "ranked_donors": [(d, best[d].receiver, best[d].ratio) for d in ranked],
        "status": status,
    }

# Default donor exposure split (since CSV has no subsegment exposures)
DEFAULT_DONOR_SPLIT_OF_TOTAL_ASSETS = {
    # Segment B donors
    "B1_SME_TERM": 0.06,
    "B1_MIDCORP_NONIG": 0.04,
    "B1_EM_CORP": 0.02,
    "B1_CRE_NON_HVCRE": 0.03,
}


# Optional per-bank donor split columns in the input CSV (percent of total assets).
# If present, these override the default donor split *per bank* (missing values fall back to defaults).
DONOR_SPLIT_COLS_PCT = {
    "B1_SME_TERM": "SME_term",
    "B1_MIDCORP_NONIG": "MidCorp_nonIG",
    "B1_EM_CORP": "EM_corporates",
    "B1_CRE_NON_HVCRE": "CRE_non_HVCRE",
}

def donor_split_from_row(row: pd.Series, default_split: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Return donor split (fractions of total assets) for one bank row.

    - Uses DEFAULT_DONOR_SPLIT_OF_TOTAL_ASSETS unless overridden by CSV % columns.
    - Each CSV % value is interpreted as percent (e.g., 6 -> 6% -> 0.06).
    - Missing/NaN/negative values fall back to defaults for that donor.
    """
    base = dict(default_split or DEFAULT_DONOR_SPLIT_OF_TOTAL_ASSETS)
    for donor, col in DONOR_SPLIT_COLS_PCT.items():
        if col in row.index:
            v = row.get(col)
            try:
                v = float(v)
            except Exception:
                v = np.nan
            if np.isfinite(v) and v >= 0:
                base[donor] = v / 100.0
    return base

def build_donor_exposures_from_total_assets(
    total_assets_eur_bn: float,
    donor_split: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    split = donor_split or DEFAULT_DONOR_SPLIT_OF_TOTAL_ASSETS
    return {k: float(total_assets_eur_bn) * float(v) for k, v in split.items()}


# ============================================================
# ---------------- Helpers (legacy) ----------------
# ============================================================
def rwa_to_assets(rwa_eur_bn: float, rwa_density_pct: float) -> float:
    density = max(float(rwa_density_pct), 1e-6) / 100.0
    return float(rwa_eur_bn) / density


def simulate_offload(years: int, banks_df: pd.DataFrame, scenarios_bps: dict, srt_efficiencies: List[float]) -> pd.DataFrame:
    """
    Per-bank version of the 'simple offload' module.
    Assumes CET1 capital is fixed: C = CET1_ratio * RWA.
    """
    rows = []
    for _, b in banks_df.iterrows():
        bank = b["Bank Name"]
        R = float(b["Total RWA (EUR bn)"])
        cet1_ratio = float(b["CET1 Ratio (%)"]) / 100.0
        C = cet1_ratio * R
        d = float(b["RWA Density (%)"])

        for sc_name, adv_bps in scenarios_bps.items():
            delta = float(adv_bps) / 10000.0
            target = cet1_ratio + delta

            # R_eff = C / target, effective reduction = R - R_eff
            R_eff = C / target if target > 0 else np.nan
            eff_red = max(R - R_eff, 0.0) if np.isfinite(R_eff) else np.nan

            for eff in srt_efficiencies:
                eff = float(eff)
                gross_rwa = (eff_red / eff) if (eff > 0 and np.isfinite(eff_red)) else np.nan
                share = (gross_rwa / R) if (R > 0 and np.isfinite(gross_rwa)) else np.nan

                ann_rwa = gross_rwa / years if np.isfinite(gross_rwa) else np.nan
                gross_ast = rwa_to_assets(gross_rwa, d) if np.isfinite(gross_rwa) else np.nan
                ann_ast = gross_ast / years if np.isfinite(gross_ast) else np.nan

                rows.append({
                    "Bank": bank,
                    "Country": b.get("Country", ""),
                    "Region": b.get("Region", ""),
                    "Reporting Period": b.get("Reporting Period", ""),
                    "Scenario": sc_name,
                    "US_CET1_Advantage_bps": float(adv_bps),
                    "SRT_Efficiency": f"{round(eff * 100):.0f}%",
                    "Years": int(years),

                    "Current_CET1_Ratio_pct": round(cet1_ratio * 100, 2),
                    "Target_CET1_Ratio_pct": round(target * 100, 2),

                    "Total_RWA_EUR_bn": R,
                    "CET1_Capital_EUR_bn": round(C, 3),
                    "RWA_Density_pct": d,

                    # Total (horizon)
                    "Effective_RWA_Reduction_EUR_bn_Tot": round(eff_red, 3) if np.isfinite(eff_red) else np.nan,
                    "Gross_RWA_Offload_EUR_bn_Tot": round(gross_rwa, 3) if np.isfinite(gross_rwa) else np.nan,
                    "Gross_RWA_Offload_pct_of_RWA_Tot": round(share * 100, 3) if np.isfinite(share) else np.nan,
                    "Gross_Assets_Offloaded_EUR_bn_Tot": round(gross_ast, 3) if np.isfinite(gross_ast) else np.nan,

                    # Annualized
                    "Gross_RWA_Offload_EUR_bn_Yr": round(ann_rwa, 3) if np.isfinite(ann_rwa) else np.nan,
                    "Gross_RWA_Offload_pct_of_RWA_Yr": round((ann_rwa / R) * 100, 3) if (R > 0 and np.isfinite(ann_rwa)) else np.nan,
                    "Gross_Assets_Offloaded_EUR_bn_Yr": round(ann_ast, 3) if np.isfinite(ann_ast) else np.nan,
                })

    return pd.DataFrame(rows)


def compute_roe_delta_transitions_greedy(
    sim_df: pd.DataFrame,
    banks_df: pd.DataFrame,
    util: float,
    override_srt_cost_bp: float | None = None,
    override_tax_rate: float | None = None,
    donor_split_override_by_bank: Optional[Dict[str, Dict[str, float]]] = None,
    donor_availability_pct_by_donor: Optional[Dict[str, float]] = None,
    require_exact_target: bool = False,
    target_tolerance_pct: float = 0.5,
) -> pd.DataFrame:
    """
    ΔROE module (transition-based).

    Workflow:
    - Take Gross_RWA_Offload (total horizon) and annualize it
    - Allocate annual RWA reduction across donor->receiver transitions (greedy)
      using only cells with (ΔRWA < 0 and ΔSpread > 0)
    - Convert each transition to profit using the corresponding Δspread (cell),
      and apply the legacy SRT-cost lever penalty logic
    - Sum profit contributions -> Addl_profit_Yr
    - Compute ROE uplift in legacy way: Addl_profit_Yr / CET1_Capital
    """
    bmap = banks_df.set_index("Bank Name")

    # Per-bank parameter maps
    srt_cost_pct = bmap["SRT Cost (%)"].to_dict()
    tax_pct = bmap["Effective Tax Rate (%)"].to_dict()
    assets_total = bmap["Total Assets (EUR bn)"].to_dict()

    DEFAULT_SRT_COST = 0.2  # percent

    df = sim_df.copy()

    # SRT cost and tax series (bank-specific unless overridden)
    sc = (df["Bank"].map(srt_cost_pct).fillna(DEFAULT_SRT_COST) / 100.0)  # decimal
    tx = (df["Bank"].map(tax_pct).fillna(0.0) / 100.0)  # decimal

    if override_srt_cost_bp is not None:
        sc = float(override_srt_cost_bp) / 10000.0
    if override_tax_rate is not None:
        tx = float(override_tax_rate)
    # Annual RWA reduction target for the transition engine:
    # Use *effective* RWA reduction (as in the original version before the gross-target change).
    gross = df["Gross_RWA_Offload_EUR_bn_Tot"].clip(lower=0)

    # Legacy lever term (gross/effective) kept so economics remain comparable
    eff = df["Effective_RWA_Reduction_EUR_bn_Tot"].clip(lower=0)
    df["gross_eff_lever"] = np.where(eff > 0, gross / eff, 1.0)
    lever_penalty = np.maximum(df["gross_eff_lever"] - 1.0, 0.0)

    years = df["Years"].clip(lower=1)
    eff_per_year = (eff / years).to_numpy(dtype=float)  # annual RWA reduction target (effective)

    # outputs
    addl_profit_yr = []
    rwa_red_yr_achieved = []
    exposure_used_yr = []
    assets_redeploy_used_yr = []
    status_list = []

    # audit trail (optional)
    audit_rows: List[Dict[str, object]] = []

    for i, row in df.iterrows():
        bank = row["Bank"]
        rwa_target_yr = float(eff_per_year[i]) if np.isfinite(eff_per_year[i]) else 0.0

        total_assets_bn = float(assets_total.get(bank, np.nan))
        if not np.isfinite(total_assets_bn) or total_assets_bn <= 0:
            addl_profit_yr.append(np.nan)
            rwa_red_yr_achieved.append(np.nan)
            exposure_used_yr.append(np.nan)
            assets_redeploy_used_yr.append(np.nan)
            status_list.append("MISSING_TOTAL_ASSETS")
            continue

        donor_split = None
        # Highest precedence: explicit overrides passed in by caller
        if donor_split_override_by_bank and bank in donor_split_override_by_bank:
            donor_split = donor_split_override_by_bank[bank]
        else:
            # Next: use per-bank donor split % columns from the CSV (if present),
            # falling back to DEFAULT_DONOR_SPLIT_OF_TOTAL_ASSETS for missing donors.
            if bank in bmap.index:
                donor_split = donor_split_from_row(bmap.loc[bank])

        donor_expo = build_donor_exposures_from_total_assets(total_assets_bn, donor_split)

        # Apply SRT-eligible availability caps (percent of donor assets available for SRT)
        # Values are expected in percent (e.g., 20 -> 20% -> 0.20). Missing donors default to 100%.
        if donor_availability_pct_by_donor:
            for _d in list(donor_expo.keys()):
                pct = donor_availability_pct_by_donor.get(_d)
                if pct is None:
                    continue
                try:
                    pct_f = float(pct)
                except Exception:
                    continue
                if np.isfinite(pct_f):
                    donor_expo[_d] = donor_expo[_d] * max(min(pct_f, 100.0), 0.0) / 100.0

        # Parse SRT efficiency string like "75%" -> 0.75
        srt_eff_str = str(row.get("SRT_Efficiency", "")).replace("%", "").strip()
        try:
            srt_eff_i = float(srt_eff_str) / 100.0
        except ValueError:
            srt_eff_i = 0.0

        # Bank-specific (or overridden) SRT cost in decimal
        sc_i_pre = float(sc.iloc[i]) if hasattr(sc, "iloc") else float(sc)

        alloc_out = greedy_allocate_rwa_reduction(rwa_target_yr, donor_expo, srt_eff_i, sc_i_pre)
        achieved = float(alloc_out["total_rwa_reduction_eur_bn"])
        expo_used = float(alloc_out["total_exposure_used_eur_bn"])
        assets_redeploy_used = float(alloc_out["total_assets_redeploy_used_eur_bn"])
        status = str(alloc_out["status"])

        # target compliance checks
        if rwa_target_yr > 0:
            gap_pct = abs(achieved - rwa_target_yr) / rwa_target_yr * 100.0 if achieved > 0 else 100.0
        else:
            gap_pct = 0.0

        if require_exact_target and (abs(achieved - rwa_target_yr) > 1e-9):
            status = "EXACT_TARGET_NOT_MET"

        if (not require_exact_target) and (rwa_target_yr > 0) and (gap_pct > target_tolerance_pct):
            status = f"{status}_OUTSIDE_TOL({gap_pct:.2f}%)"

        # compute profits from allocations (transition-based)
        sc_i = float(sc.iloc[i]) if hasattr(sc, "iloc") else float(sc)
        tx_i = float(tx.iloc[i]) if hasattr(tx, "iloc") else float(tx)
        lever_i = float(lever_penalty.iloc[i]) if hasattr(lever_penalty, "iloc") else float(lever_penalty)

        profit = 0.0
        for a in alloc_out["allocations"]:
            # delta_s_eff_dec is precomputed in the allocator using:
            #   (-ΔRWA_pp)/100 * SRT_efficiency * (abs net spread of receiver) - 2 * SRT_cost
            contrib = a.exposure_used_eur_bn * a.delta_s_eff_dec * (1.0 - tx_i) * float(util)
            profit += contrib

            audit_rows.append({
                "Bank": bank,
                "Scenario": row["Scenario"],
                "SRT_Efficiency": row["SRT_Efficiency"],
                "Years": int(row["Years"]),
                "Donor": a.donor,
                "Receiver": a.receiver,
                "Exposure_used_EUR_bn_Yr": a.exposure_used_eur_bn,
                "RWA_reduction_EUR_bn_Yr": a.rwa_reduction_eur_bn,
                "Delta_RWA_pp": a.delta_rwa_pp,
                "Abs_net_spread_bps": a.abs_net_spread_bps,
                "SRT_cost_dec": sc_i,
                "Delta_s_eff_dec": a.delta_s_eff_dec,
                "Profit_contrib_EUR_bn_Yr": contrib,
                "Status": status,
            })

        addl_profit_yr.append(profit)
        rwa_red_yr_achieved.append(achieved)
        exposure_used_yr.append(expo_used)
        assets_redeploy_used_yr.append(assets_redeploy_used)
        status_list.append(status)

    df["RWA_reduction_achieved_Yr"] = rwa_red_yr_achieved
    df["Assets_redeploy_used_Yr"] = assets_redeploy_used_yr  # receiver-side assets redeployed (implied by receiver risk weights)
    df["Addl_profit_Yr"] = addl_profit_yr
    df["Transition_status"] = status_list

    cap = df["CET1_Capital_EUR_bn"].clip(lower=1e-6)
    df["ROE_delta_bp"] = np.round((df["Addl_profit_Yr"] / cap) * 10000.0, 3)

    out = df[[
        "Bank", "Country", "Region", "Reporting Period",
        "Scenario", "SRT_Efficiency", "Years",
        "CET1_Capital_EUR_bn",
        "RWA_reduction_achieved_Yr",
        "Assets_redeploy_used_Yr",
        "ROE_delta_bp",
        "gross_eff_lever",
        "Transition_status",
    ]].copy()

    out.attrs["allocations_audit_df"] = pd.DataFrame(audit_rows)
    return out


def compute_sri(sim_df: pd.DataFrame, banks_df: pd.DataFrame) -> pd.DataFrame:
    """
    SRI = 100 * (Gross Assets Offloaded / Bank Total Assets)
    Uses 'Total Assets (EUR bn)' from the bank input file as denominator.
    """
    assets_total = banks_df.set_index("Bank Name")["Total Assets (EUR bn)"].to_dict()
    df = sim_df.copy()
    df["Bank_Assets_Total"] = df["Bank"].map(assets_total)
    denom = df["Bank_Assets_Total"].clip(lower=1e-6)

    df["SRI"] = np.round(100.0 * (df["Gross_Assets_Offloaded_EUR_bn_Tot"] / denom), 3)

    return df[[
        "Bank", "Country", "Region", "Reporting Period",
        "Scenario", "SRT_Efficiency",
        "SRI", "Gross_Assets_Offloaded_EUR_bn_Tot", "Bank_Assets_Total"
    ]]


def to_xlsx_bytes(sim_df: pd.DataFrame, roe_df: pd.DataFrame, sri_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        sim_df.to_excel(writer, sheet_name="Simulation", index=False)
        roe_df.to_excel(writer, sheet_name="ROE", index=False)
        sri_df.to_excel(writer, sheet_name="SRI", index=False)

        # Optional: include allocations audit if present
        audit = roe_df.attrs.get("allocations_audit_df")
        if isinstance(audit, pd.DataFrame) and not audit.empty:
            audit.to_excel(writer, sheet_name="ROE_Transitions", index=False)

    return buf.getvalue()


def make_portfolio_row(banks_sel: pd.DataFrame) -> pd.DataFrame:
    df = banks_sel.copy()

    # Fallbacks for sparse columns (per-bank first)
    df["Net Spread (%)"] = df["Net Spread (%)"].fillna(2.5)
    df["SRT Cost (%)"] = df["SRT Cost (%)"].fillna(0.2)
    df["Effective Tax Rate (%)"] = df["Effective Tax Rate (%)"].fillna(0.0)

    # CET1 capital per bank (bn EUR)
    df["_cet1_cap_bn"] = (df["CET1 Ratio (%)"] / 100.0) * df["Total RWA (EUR bn)"]

    total_assets = float(df["Total Assets (EUR bn)"].sum())
    total_rwa = float(df["Total RWA (EUR bn)"].sum())
    total_cet1_cap = float(df["_cet1_cap_bn"].sum())

    # Derived portfolio ratios
    cet1_ratio_pct = (total_cet1_cap / total_rwa) * 100.0 if total_rwa > 0 else np.nan
    rwa_density_pct = (total_rwa / total_assets) * 100.0 if total_assets > 0 else np.nan

    # Asset-weighted averages for spread/cost/tax
    w = df["Total Assets (EUR bn)"].to_numpy(dtype=float)
    wsum = np.nansum(w)

    def wavg(x):
        x = np.asarray(x, dtype=float)
        return float(np.nansum(x * w) / wsum) if wsum > 0 else np.nan

    spread_pct = wavg(df["Net Spread (%)"].to_numpy())
    srt_cost_pct = wavg(df["SRT Cost (%)"].to_numpy())
    tax_pct = wavg(df["Effective Tax Rate (%)"].to_numpy())

    
    # If donor split % columns exist, aggregate them to a portfolio-level % (asset-weighted).
    donor_pct_fields: Dict[str, float] = {}
    if any(c in df.columns for c in DONOR_SPLIT_COLS_PCT.values()):
        ta = df["Total Assets (EUR bn)"].to_numpy(dtype=float)
        ta_sum = float(np.nansum(ta))
        for donor, col in DONOR_SPLIT_COLS_PCT.items():
            if col not in df.columns:
                continue
            pct = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 100.0
            expo = float(np.nansum(ta * pct))
            donor_pct_fields[col] = (expo / ta_sum * 100.0) if ta_sum > 0 else np.nan

    portfolio = pd.DataFrame([{
        "Bank Name": "PORTFOLIO (Selected banks)",
        "Country": "—",
        "Region": "—",
        "Reporting Period": "—",
        "Total Assets (EUR bn)": total_assets,
        "Total RWA (EUR bn)": total_rwa,
        "CET1 Ratio (%)": cet1_ratio_pct,
        "RWA Density (%)": rwa_density_pct,
        "Net Spread (%)": spread_pct,
        "SRT Cost (%)": srt_cost_pct,
        "Effective Tax Rate (%)": tax_pct,
        **donor_pct_fields
    }])

    return portfolio


# ============================================================
# ---------------- Streamlit App ----------------
# ============================================================
st.set_page_config(page_title="Bank-specific Offload Simulation", layout="wide")
st.title("Bank-specific Offload Simulation")

# ---- Load bank input data automatically ----
DATA_PATH = "52_banks_full_results.csv"

try:
    banks = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"""
        ❌ Bank input file not found.

        Expected file:
        {DATA_PATH}

        Please make sure the CSV is located in the same folder as app.py.
        """
    )
    st.stop()

required_cols = [
    "Bank Name", "Total Assets (EUR bn)", "Total RWA (EUR bn)", "CET1 Ratio (%)",
    "RWA Density (%)", "Net Spread (%)", "SRT Cost (%)", "Effective Tax Rate (%)"
]
missing = [c for c in required_cols if c not in banks.columns]
if missing:
    st.error(f"CSV is missing required columns: {missing}")
    st.stop()

# Clean / ensure types
banks = banks.copy()
banks["Bank Name"] = banks["Bank Name"].astype(str)

# ---- Main layout: charts left, controls right ----
left_col, right_col = st.columns([8, 4], gap="large")

with right_col:
    st.header("Global Controls")

    years = st.slider("Horizont (Jahre)", min_value=1, max_value=10, value=5, step=1)

    # Filters
    if "Country" in banks.columns:
        countries = sorted([c for c in banks["Country"].dropna().unique().tolist()])
        sel_countries = st.multiselect("Filter: Country", countries, default=countries)
        banks_f = banks[banks["Country"].isin(sel_countries)].copy()
    #else:
    #    banks_f = banks.copy()

    if "Region" in banks.columns:
        regions = sorted([r for r in banks_f["Region"].dropna().unique().tolist()])
        sel_regions = st.multiselect("Filter: Region", regions, default=regions)
        banks_f = banks_f[banks_f["Region"].isin(sel_regions)].copy()

    # Bank selection
    bank_list = sorted(banks_f["Bank Name"].unique().tolist())
    default_sel = bank_list[: min(8, len(bank_list))]
    selected_banks = st.multiselect("Banks (toggle on/off)", bank_list, default=default_sel)

    st.markdown("---")
    st.subheader("Transition engine controls")

    st.markdown("**SRT-eligible share of donor assets (% of each donor bucket, max 20%)**")
    avail_sme = st.slider("SME term available for SRT (%)", 0, 20, 20, 1)
    avail_mid = st.slider("Mid-corp non-IG available for SRT (%)", 0, 20, 20, 1)
    avail_em = st.slider("EM corporates available for SRT (%)", 0, 20, 20, 1)
    avail_cre = st.slider("CRE non-HVCRE available for SRT (%)", 0, 20, 20, 1)

    donor_availability_pct = {
        "B1_SME_TERM": float(avail_sme),
        "B1_MIDCORP_NONIG": float(avail_mid),
        "B1_EM_CORP": float(avail_em),
        "B1_CRE_NON_HVCRE": float(avail_cre),
    }

    require_exact = st.checkbox("Require exact annual RWA target", value=False)
    tol_pct = st.slider("Target tolerance (%)", 0.0, 5.0, 0.5, 0.1)

    show_audit = st.checkbox("Show transition audit table", value=False)

    # Placeholder for capacity indicator (filled after model run)
    capacity_placeholder = st.empty()

    st.markdown("---")
    st.subheader("Offload Display")
    metric = st.radio("Metric", ["% der RWA", "Assets (EUR bn)", "RWA (EUR bn)"], index=1)
    agg = st.radio("Aggregation", ["Total (Horizont)", "jährlich"], index=0)

    assets_method = st.radio(
        "Assets offload method",
        ["Simple (RWA density)", "Transition-based (donor RW)"],
        index=0,
        help=(
            "Simple uses Gross_RWA_Offload converted to assets via bank RWA density. "
            "Transition-based uses the transition engine's achieved RWA reduction per year, "
            "converted to assets via the receiver risk weights and summed across contributions."
        ),
    )

    st.markdown("---")
    st.subheader("Scenarios (US advantage in bps)")
    scen_enable = st.multiselect("Enabled scenarios", ["S1", "S2", "S3", "Custom"], default=["S1", "S2", "S3"])
    s1_bps = st.slider("S1 (bp)", 0, 300, 80, 5)
    s2_bps = st.slider("S2 (bp)", 0, 300, 168, 5)
    s3_bps = st.slider("S3 (bp)", 0, 400, 218, 5)
    cust_bps = st.slider("Custom (bp)", 0, 400, 120, 5)

    st.subheader("SRT efficiencies")
    eff1 = st.slider("A", 0.0, 1.0, 0.60, 0.01)
    eff2 = st.slider("B", 0.0, 1.0, 0.75, 0.01)
    eff3 = st.slider("C", 0.0, 1.0, 0.90, 0.01)
    eff4 = st.slider("D (Custom)", 0.0, 1.0, 0.80, 0.01)

    st.markdown("---")
    st.subheader("Complex module (ΔROE)")
    util = st.slider("Utilization (%)", 50, 100, 85, 1) / 100.0

    override = st.checkbox("Override bank-specific cost/tax with global values", value=False)
    override_srt_cost_bp = override_tax_rate = None
    if override:
        override_srt_cost_bp = st.slider("Global SRT Cost (bp)", 0, 100, 20, 5)
        override_tax_rate = st.slider("Global Tax rate (%)", 0, 50, 25, 1) / 100.0


# Validate selections
if not selected_banks:
    st.error("Please select at least one bank in the sidebar.")
    st.stop()

banks_sel = banks_f[banks_f["Bank Name"].isin(selected_banks)].copy()

# Build scenarios dict
scenarios = {}
if "S1" in scen_enable:
    scenarios["S1_Partial_US_Easing"] = s1_bps
if "S2" in scen_enable:
    scenarios["S2_Full_US_Easing"] = s2_bps
if "S3" in scen_enable:
    scenarios["S3_US_Easing_plus_EU_Tightening"] = s3_bps
if "Custom" in scen_enable:
    scenarios["Custom"] = cust_bps
if not scenarios:
    st.error("Please enable at least one scenario.")
    st.stop()

effs = sorted({round(e, 4) for e in [eff1, eff2, eff3, eff4]})

portfolio_df = make_portfolio_row(banks_sel)

# Run model for portfolio
sim_port = simulate_offload(years, portfolio_df, scenarios, effs)
roe_port = compute_roe_delta_transitions_greedy(
    sim_port,
    portfolio_df,
    util=util,
    override_srt_cost_bp=override_srt_cost_bp,
    override_tax_rate=override_tax_rate,
    require_exact_target=require_exact,
    target_tolerance_pct=tol_pct,
    donor_availability_pct_by_donor=donor_availability_pct,
)
sri_port = compute_sri(sim_port, portfolio_df)

# ---- Run model ----
sim_df = simulate_offload(years, banks_sel, scenarios, effs)
roe_df = compute_roe_delta_transitions_greedy(
    sim_df,
    banks_sel,
    util=util,
    override_srt_cost_bp=override_srt_cost_bp,
    override_tax_rate=override_tax_rate,
    require_exact_target=require_exact,
    target_tolerance_pct=tol_pct,
    donor_availability_pct_by_donor=donor_availability_pct,
)
sri_df = compute_sri(sim_df, banks_sel)


# ============================================================
# Sidebar capacity indicator (lights up when target not met)
# ============================================================
def _is_insufficient_transition(status: object) -> bool:
    """Return True if the transition engine could not meet the annual RWA target."""
    s = str(status) if status is not None else ""
    if s == "OK":
        return False
    # Anything else implies infeasibility / non-compliance with target (incl. tolerance / exact).
    return True


insufficient_rows = roe_df[roe_df["Transition_status"].apply(_is_insufficient_transition)]
insufficient_rows_port = roe_port[roe_port["Transition_status"].apply(_is_insufficient_transition)]
insufficient_any = (not insufficient_rows.empty) or (not insufficient_rows_port.empty)

with capacity_placeholder.container():
    st.markdown("---")
    st.subheader("SRT Capacity Check")

    if insufficient_any:
        st.error("🚨 Insufficient SRT-eligible assets: target annual RWA reduction cannot be met for all combinations.")
        st.caption(
            "Reduce the risk by increasing donor availability sliders or relaxing the target tolerance/exactness settings."
        )

        # A button-like interaction to reveal details
        if st.button("Show failing combinations"):
            cols = [
                "Bank",
                "Scenario",
                "SRT_Efficiency",
                "Years",
                "Transition_status",
                "RWA_reduction_achieved_Yr",
            ]
            st.dataframe(insufficient_rows[cols], use_container_width=True, height=240)
            st.dataframe(insufficient_rows_port[cols], use_container_width=True, height=160)
    else:
        st.success("✅ Capacity sufficient: target annual RWA reduction is achievable for all displayed combinations.")

# ---- Optional alternative assets-offload measure (transition-based) ----
def _attach_transition_based_assets(sim: pd.DataFrame, roe: pd.DataFrame) -> pd.DataFrame:
    """Attach transition-based asset offload columns to the simple simulation output.

    Transition-based (donor RW) assets offloaded are computed as:

      Step 1: aggregate annual RWA contributions by donor:
              RWA_d = sum_{allocations in row and donor d} RWA_reduction_EUR_bn_Yr
      Step 2: convert each donor RWA to assets using donor RW:
              Assets_net = sum_d (RWA_d / Donor_RW_d)
      Step 3: gross-up by dividing by efficiency:
              Assets_gross = Assets_net / efficiency

    The result is attached as:
      - Assets_Offloaded_Transition_EUR_bn_Yr
      - Assets_Offloaded_Transition_EUR_bn_Tot
    """
    sim = sim.copy()

    # Ensure we don't create duplicate columns when merging (avoid _x/_y suffixes)
    sim = sim.drop(columns=[
        "Assets_Offloaded_Transition_EUR_bn_Yr",
        "Assets_Offloaded_Transition_EUR_bn_Tot",
        "Assets_Offloaded_Transition_net_EUR_bn_Yr",
    ], errors="ignore")

    if roe is None or roe.empty:
        return sim

    audit = roe.attrs.get("allocations_audit_df")
    if not isinstance(audit, pd.DataFrame) or audit.empty:
        return sim

    key = ["Bank", "Scenario", "SRT_Efficiency", "Years"]

    # Aggregate RWA contributions by donor within each (Bank, Scenario, Efficiency, Years)
    grp = (
        audit.groupby(key + ["Donor"], dropna=False)["RWA_reduction_EUR_bn_Yr"]
        .sum()
        .reset_index()
    )

    # Map donor RW and convert donor RWA to assets
    grp["Donor_RW"] = grp["Donor"].map(DONOR_RISK_WEIGHT).astype(float)
    grp.loc[grp["Donor_RW"] <= 0, "Donor_RW"] = np.nan
    grp["Assets_net_from_donor"] = grp["RWA_reduction_EUR_bn_Yr"] / grp["Donor_RW"]

    # Sum assets across donors per row-key
    assets_net = (
        grp.groupby(key, dropna=False)["Assets_net_from_donor"]
        .sum()
        .reset_index()
        .rename(columns={"Assets_net_from_donor": "Assets_Offloaded_Transition_net_EUR_bn_Yr"})
    )

    # Parse efficiency and gross-up
    eff_dec = assets_net["SRT_Efficiency"].astype(str).str.replace("%", "", regex=False).astype(float) / 100.0
    eff_dec = eff_dec.replace(0.0, np.nan)
    assets_net["Assets_Offloaded_Transition_EUR_bn_Yr"] = assets_net["Assets_Offloaded_Transition_net_EUR_bn_Yr"] / eff_dec
    assets_net["Assets_Offloaded_Transition_EUR_bn_Tot"] = assets_net["Assets_Offloaded_Transition_EUR_bn_Yr"] * assets_net["Years"].astype(float)

    # Merge back to sim
    sim = sim.merge(
        assets_net[key + ["Assets_Offloaded_Transition_EUR_bn_Yr", "Assets_Offloaded_Transition_EUR_bn_Tot"]],
        on=key,
        how="left",
    )
    return sim


sim_df = _attach_transition_based_assets(sim_df, roe_df)
sim_port = _attach_transition_based_assets(sim_port, roe_port)

# ---- Charts ----
with left_col:
    st.subheader("1) Offload (Simple)")

    if metric == "% der RWA" and agg == "Total (Horizont)":
        yv, yl = "Gross_RWA_Offload_pct_of_RWA_Tot", "% der RWA (Total)"
    elif metric == "% der RWA" and agg == "jährlich":
        yv, yl = "Gross_RWA_Offload_pct_of_RWA_Yr", "% der RWA (jährlich)"
    elif metric == "Assets (EUR bn)" and agg == "Total (Horizont)":
        if assets_method == "Transition-based (donor RW)":
            yv, yl = "Assets_Offloaded_Transition_EUR_bn_Tot", "Assets-Offload (EUR bn, Total, transition-based)"
        else:
            yv, yl = "Gross_Assets_Offloaded_EUR_bn_Tot", "Assets-Offload (EUR bn, Total)"
    elif metric == "Assets (EUR bn)" and agg == "jährlich":
        if assets_method == "Transition-based (donor RW)":
            yv, yl = "Assets_Offloaded_Transition_EUR_bn_Yr", "Assets-Offload (EUR bn, jährlich, transition-based)"
        else:
            yv, yl = "Gross_Assets_Offloaded_EUR_bn_Yr", "Assets-Offload (EUR bn, jährlich)"
    elif metric == "RWA (EUR bn)" and agg == "Total (Horizont)":
        yv, yl = "Gross_RWA_Offload_EUR_bn_Tot", "RWA-Offload (EUR bn, Total)"
    else:
        yv, yl = "Gross_RWA_Offload_EUR_bn_Yr", "RWA-Offload (EUR bn, jährlich)"

    fig1 = px.bar(
        sim_df,
        x="Scenario",
        y=yv,
        color="SRT_Efficiency",
        barmode="group",
        facet_col="Bank",
        facet_col_wrap=3,
        labels={yv: yl, "Scenario": "", "SRT_Efficiency": "SRT-Effizienz"},
        title="Erforderlicher Offload nach Szenario & Effizienz (pro Bank)"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")
    st.subheader("2) Offload Complex (ΔROE & SRI)")

    fig2 = px.bar(
        roe_df,
        x="Scenario",
        y="ROE_delta_bp",
        color="SRT_Efficiency",
        barmode="group",
        facet_col="Bank",
        facet_col_wrap=3,
        labels={"ROE_delta_bp": "ΔROE (bp p.a.)", "Scenario": "", "SRT_Efficiency": "SRT-Effizienz"},
        title="ΔROE (bp p.a.) – Transition-based redeployment (greedy)"
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(
        sri_df,
        x="Scenario",
        y="SRI",
        color="SRT_Efficiency",
        barmode="group",
        facet_col="Bank",
        facet_col_wrap=3,
        labels={"SRI": "SRI (%)", "Scenario": "", "SRT_Efficiency": "SRT-Effizienz"},
        title="System Risk Index (SRI) – Anteil ausgelagerter Assets (pro Bank)"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        """
        <div style="background:#f7f7f7;border:1px solid #ddd;border-radius:6px;padding:10px;margin-top:8px;">
          <h5 style="margin:0 0 6px 0;">Szenarienbeschreibung</h5>
          <ul style="margin:0; padding-left:18px;">
            <li><b>S1 – Moderate US-Deregulierung:</b> Teilweise Lockerung der US-Stresstests (~80 bp CET1-Vorteil).</li>
            <li><b>S2 – Volle US-Deregulierung:</b> Umfassende Entlastung (≈168 bp CET1-Vorteil).</li>
            <li><b>S3 – US-Deregulierung + EU-Gegenwind:</b> Zusätzlicher EU-Impact, gesamt ≈218 bp.</li>
            <li><b>Custom – Nutzerdefiniert:</b> frei einstellbarer CET1-Vorteil in bp.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.subheader("PORTFOLIO (aggregate across selected banks)")

    figP1 = px.bar(
        sim_port,
        x="Scenario",
        y=yv,
        color="SRT_Efficiency",
        barmode="group",
        labels={yv: yl, "Scenario": "", "SRT_Efficiency": "SRT-Effizienz"},
        title="Erforderlicher Offload – Portfolio"
    )
    st.plotly_chart(figP1, use_container_width=True)

    figP2 = px.bar(
        roe_port,
        x="Scenario",
        y="ROE_delta_bp",
        color="SRT_Efficiency",
        barmode="group",
        labels={"ROE_delta_bp": "ΔROE (bp p.a.)", "Scenario": "", "SRT_Efficiency": "SRT-Effizienz"},
        title="ΔROE (bp p.a.) – Portfolio (transition-based)"
    )
    st.plotly_chart(figP2, use_container_width=True)

    figP3 = px.bar(
        sri_port,
        x="Scenario",
        y="SRI",
        color="SRT_Efficiency",
        barmode="group",
        labels={"SRI": "SRI (%)", "Scenario": "", "SRT_Efficiency": "SRT-Effizienz"},
        title="SRI (%) – Portfolio"
    )
    st.plotly_chart(figP3, use_container_width=True)

    if show_audit:
        st.markdown("---")
        st.subheader("Transition Audit (per-row allocations)")
        audit_df = roe_df.attrs.get("allocations_audit_df")
        if isinstance(audit_df, pd.DataFrame) and not audit_df.empty:
            st.dataframe(audit_df, use_container_width=True, height=420)
        else:
            st.info("No audit rows (likely no eligible transitions or zero targets).")


# ---- XLSX Export ----
xlsx_bytes = to_xlsx_bytes(sim_df, roe_df, sri_df)
st.download_button(
    label="XLSX herunterladen",
    data=xlsx_bytes,
    file_name=f"offload_banks_{pd.Timestamp.today().date()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
