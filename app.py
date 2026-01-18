import io
import re
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

# Global constant: total assets column name
TOTAL_ASSETS_COL = "Total Assets (EUR bn)"


import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
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


def _eligible_cells_by_donor(
    donors: List[str],
    srt_efficiency: float,
    srt_cost_dec: float,
    delta_rwa_pp: Dict[Tuple[str, str], float],
    delta_spread_bps: Dict[Tuple[str, str], float],
) -> Dict[str, List[Cell]]:
    """Return all *eligible* receiver cells per donor.

    Eligibility (transition permitted):
      - DELTA_RWA_PP_B cell value must be < -1.0
      - DELTA_SPREAD_BPS_B cell value must be > 0

    Note: We do **no prioritization** across receivers. Allocation will split
    a donor's used exposure equally across all eligible receivers.

    We still compute delta_s_eff_dec and ratio for reporting/auditing.
    """
    cells: Dict[str, List[Cell]] = {d: [] for d in donors}
    eff = float(srt_efficiency)

    for (d, r), drwa in delta_rwa_pp.items():
        if d not in donors:
            continue

        dspr = delta_spread_bps.get((d, r))
        if dspr is None:
            continue

        donor_rw = float(DONOR_RISK_WEIGHT.get(d, 0.0))
        receiver_rw = float(RECEIVER_RISK_WEIGHT.get(r, 0.0))
        abs_spread_bps = float(ABS_NET_SPREAD_BPS_BY_B2.get(r, 0.0))

        if donor_rw <= 0 or receiver_rw <= 0:
            continue

        # permitted transitions:
        if not (float(drwa) < -1.0 and float(dspr) > 0.0):
            continue

        # delta_rwa_density stays "as-is" (you use quotient-like values)
        delta_rwa_density = (-float(drwa))

        # economics term (bps -> decimal)
        srt_cost_bp = float(srt_cost_dec) * 10000.0
        delta_s_eff_bps = delta_rwa_density * eff * abs_spread_bps - 2.0 * srt_cost_bp
        delta_s_eff_dec = delta_s_eff_bps / 10000.0

        rwa_red_per_eur = donor_rw * eff
        ratio = (delta_s_eff_dec / rwa_red_per_eur) if rwa_red_per_eur > 0 else 0.0

        cells[d].append(
            Cell(
                donor=d,
                receiver=r,
                delta_rwa_pp=float(drwa),
                donor_risk_weight=donor_rw,
                receiver_risk_weight=receiver_rw,
                abs_net_spread_bps=abs_spread_bps,
                delta_s_eff_dec=float(delta_s_eff_dec),
                ratio=float(ratio),
            )
        )

    # drop donors with no eligible receivers
    return {d: lst for d, lst in cells.items() if lst}


def allocate_rwa_reduction_equal_receivers(
    rwa_target_eur_bn: float,
    donor_exposure_eur_bn: Dict[str, float],
    srt_efficiency: float,
    srt_cost_dec: float,
) -> Dict[str, object]:
    """Allocator (Segment B only) with *equal split across eligible receivers*.

    Changes vs earlier greedy version:
      - No prioritization across donors by profitability.
      - For each donor, exposure used is split equally across all eligible receivers.

    RWA reduction achieved per allocation line:
        rwa_reduction = exposure_used * donor_risk_weight * srt_efficiency

    Assets redeployed used per allocation line (receiver-side):
        assets_redeploy = rwa_reduction / receiver_risk_weight
    """
    donors = [d for d in B1_DONORS if donor_exposure_eur_bn.get(d, 0.0) > 0]

    cells_by_donor = _eligible_cells_by_donor(
        donors=donors,
        srt_efficiency=float(srt_efficiency),
        srt_cost_dec=float(srt_cost_dec),
        delta_rwa_pp=DELTA_RWA_PP_B,
        delta_spread_bps=DELTA_SPREAD_BPS_B,
    )

    if not cells_by_donor:
        return {
            "allocations": [],
            "total_rwa_reduction_eur_bn": 0.0,
            "total_exposure_used_eur_bn": 0.0,
            "total_assets_redeploy_used_eur_bn": 0.0,
            "remaining_rwa_target_eur_bn": float(rwa_target_eur_bn),
            "ranked_donors": [],
            "status": "NO_ELIGIBLE_TRANSITIONS",
        }

    remaining = float(rwa_target_eur_bn)
    allocs: List[Allocation] = []
    total_rwa_red = 0.0
    total_expo = 0.0
    total_assets_redeploy = 0.0

    eff = float(srt_efficiency)

    # No prioritization: iterate donors in fixed order
    for d in B1_DONORS:
        if remaining <= 0:
            break
        if d not in cells_by_donor:
            continue

        expo_avail = float(donor_exposure_eur_bn.get(d, 0.0))
        if expo_avail <= 0:
            continue

        # donor RWA reduction per EUR exposure moved
        donor_rw = float(DONOR_RISK_WEIGHT.get(d, 0.0))
        red_per_eur = donor_rw * eff
        if red_per_eur <= 0:
            continue

        expo_needed = remaining / red_per_eur
        expo_used_total = min(expo_avail, expo_needed)
        if expo_used_total <= 0:
            continue

        # Split equally across eligible receivers
        cells = cells_by_donor[d]
        k = len(cells)
        expo_each = expo_used_total / k

        for cell in cells:
            expo_used = expo_each
            rwa_red = expo_used * red_per_eur
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

        remaining -= expo_used_total * red_per_eur

    status = "OK" if remaining <= 1e-9 else "TARGET_NOT_MET"
    return {
        "allocations": allocs,
        "total_rwa_reduction_eur_bn": total_rwa_red,
        "total_exposure_used_eur_bn": total_expo,
        "total_assets_redeploy_used_eur_bn": total_assets_redeploy,
        "remaining_rwa_target_eur_bn": max(0.0, remaining),
        # kept key name for UI compatibility; now it's just donor order with receiver count
        "ranked_donors": [(d, "MULTI", len(cells_by_donor[d])) for d in B1_DONORS if d in cells_by_donor],
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


def donor_eligible_exposure_long(
    banks_df: pd.DataFrame,
    donor_availability_pct_by_donor: Optional[Dict[str, float]] = None,
    donor_split_override_by_bank: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Return long df of eligible donor exposure per bank (EUR bn).

    Eligible exposure = Total Assets * donor split * availability cap (percent).
    Availability caps are in percent (e.g. 10 -> 10%). Missing donors default to 100%.
    """
    rows: list[dict] = []
    for _, r in banks_df.iterrows():
        bank = str(r.get("Bank", ""))
        if not bank or bank == "nan":
            bank = str(r.get("Bank Name", ""))
        ta = float(r.get("Total Assets (EUR bn)", np.nan))
        if not np.isfinite(ta):
            continue
        split = donor_split_from_row(r, DEFAULT_DONOR_SPLIT_OF_TOTAL_ASSETS)
        # apply optional override dict by bank name
        if donor_split_override_by_bank and bank in donor_split_override_by_bank:
            for d, v in donor_split_override_by_bank[bank].items():
                try:
                    split[d] = float(v)
                except Exception:
                    pass
        for donor, w in split.items():
            try:
                w_f = float(w)
            except Exception:
                continue
            if not np.isfinite(w_f) or w_f <= 0:
                continue
            expo = ta * w_f
            cap_pct = 100.0
            if donor_availability_pct_by_donor and donor in donor_availability_pct_by_donor:
                try:
                    cap_pct = float(donor_availability_pct_by_donor.get(donor, 100.0))
                except Exception:
                    cap_pct = 100.0
            if np.isfinite(cap_pct):
                cap_pct = max(min(cap_pct, 100.0), 0.0)
            else:
                cap_pct = 100.0
            expo_elig = expo * cap_pct / 100.0
            rows.append({"Bank": bank, "Donor": donor, "Eligible_Exposure_EUR_bn": expo_elig})
    return pd.DataFrame(rows)

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
    apply_util_target_scaling: bool = True,
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
    rwa_target_base_yr_list = []
    rwa_target_scaled_yr_list = []
    rwa_target_scale_list = []

    # audit trail (optional)
    audit_rows: List[Dict[str, object]] = []

    for i, row in df.iterrows():
        bank = row["Bank"]
        # Base (effective) annual RWA reduction target
        rwa_target_yr_base = float(eff_per_year[i]) if np.isfinite(eff_per_year[i]) else 0.0

        # Two-step allocation:
        #   Step 1 (Redeployment): allocate the RWAs needed to support redeployment into new assets.
        #   Step 2 (CET1 uplift): allocate the RWAs required for CET1 uplift, *after* Step 1 has consumed capacity.
        #
        # Profit / ROE uplift must be computed from Step 1 only.
        util_dec = float(util) if util is not None else 0.0
        if not np.isfinite(util_dec):
            util_dec = 0.0
        util_dec = max(min(util_dec, 0.999999), 0.0)

        # Base (annual effective) CET1-uplift requirement:
        rwa_target_cet1_yr = rwa_target_yr_base

        # Redeployment requires additional RWAs; factor = util/(1-util).
        if apply_util_target_scaling and util_dec > 0:
            target_scale = (1.0 / (1.0 - util_dec)) - 1.0  # = util_dec/(1-util_dec)
            rwa_target_redeploy_yr = rwa_target_yr_base * target_scale
        else:
            target_scale = 0.0
            rwa_target_redeploy_yr = 0.0

        # Total RWA reduction volume that consumes transition capacity:
        rwa_target_total_yr = rwa_target_redeploy_yr + rwa_target_cet1_yr

        # store targets for reporting (aligned with df rows)
        rwa_target_base_yr_list.append(rwa_target_yr_base)     # CET1 portion (base)
        rwa_target_scaled_yr_list.append(rwa_target_total_yr)  # total capacity-consuming volume
        rwa_target_scale_list.append(target_scale)             # redeployment factor only

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

        # ------------------------------
        # Step 1: Redeployment allocation
        # ------------------------------
        alloc_redeploy = allocate_rwa_reduction_equal_receivers(
            rwa_target_redeploy_yr, donor_expo, srt_eff_i, sc_i_pre
        )

        achieved_redeploy = float(alloc_redeploy["total_rwa_reduction_eur_bn"])
        expo_used_redeploy = float(alloc_redeploy["total_exposure_used_eur_bn"])
        assets_redeploy_used = float(alloc_redeploy["total_assets_redeploy_used_eur_bn"])
        status_redeploy = str(alloc_redeploy["status"])

        # Net out donor exposure already consumed in Step 1
        used_by_donor = {}
        for a in alloc_redeploy["allocations"]:
            used_by_donor[a.donor] = used_by_donor.get(a.donor, 0.0) + float(a.exposure_used_eur_bn)

        donor_expo_remaining = dict(donor_expo)
        for d_k, used in used_by_donor.items():
            if d_k in donor_expo_remaining:
                donor_expo_remaining[d_k] = max(0.0, float(donor_expo_remaining[d_k]) - float(used))

        # ------------------------------
        # Step 2: CET1 uplift allocation
        # ------------------------------
        alloc_cet1 = allocate_rwa_reduction_equal_receivers(
            rwa_target_cet1_yr, donor_expo_remaining, srt_eff_i, sc_i_pre
        )

        achieved_cet1 = float(alloc_cet1["total_rwa_reduction_eur_bn"])
        expo_used_cet1 = float(alloc_cet1["total_exposure_used_eur_bn"])
        status_cet1 = str(alloc_cet1["status"])

        # Totals (capacity consumption)
        achieved_total = achieved_redeploy + achieved_cet1
        expo_used_total = expo_used_redeploy + expo_used_cet1

        # ------------------------------
        # Target compliance checks (TOTAL)
        # ------------------------------
        if rwa_target_total_yr > 0:
            gap_pct = abs(achieved_total - rwa_target_total_yr) / rwa_target_total_yr * 100.0 if achieved_total > 0 else 100.0
        else:
            gap_pct = 0.0

        # Compose a status that reflects both steps
        status = "OK"
        if (rwa_target_redeploy_yr > 0) and (status_redeploy != "OK"):
            status = f"REDEPLOY_{status_redeploy}"
        if (rwa_target_cet1_yr > 0) and (status_cet1 != "OK"):
            status = f"{status}_CET1_{status_cet1}" if status != "OK" else f"CET1_{status_cet1}"

        if require_exact_target and (abs(achieved_total - rwa_target_total_yr) > 1e-9):
            status = "EXACT_TARGET_NOT_MET"

        if (not require_exact_target) and (rwa_target_total_yr > 0) and (gap_pct > target_tolerance_pct):
            status = f"{status}_OUTSIDE_TOL({gap_pct:.2f}%)"

        # ------------------------------
        # Profit calculation (Step 1 ONLY)
        # ------------------------------
        sc_i = float(sc.iloc[i]) if hasattr(sc, "iloc") else float(sc)
        tx_i = float(tx.iloc[i]) if hasattr(tx, "iloc") else float(tx)
        lever_i = float(lever_penalty.iloc[i]) if hasattr(lever_penalty, "iloc") else float(lever_penalty)

        profit = 0.0

        # Audit + profit for Redeployment allocations only
        for a in alloc_redeploy["allocations"]:
            contrib = a.exposure_used_eur_bn * a.delta_s_eff_dec * (1.0 - tx_i)
            profit += contrib

            audit_rows.append({
                "Bank": bank,
                "Scenario": row["Scenario"],
                "SRT_Efficiency": row["SRT_Efficiency"],
                "Years": int(row["Years"]),
                "Step": "REDEPLOY",
                "RWA_target_base_EUR_bn_Yr": rwa_target_yr_base,
                "RWA_target_redeploy_EUR_bn_Yr": rwa_target_redeploy_yr,
                "RWA_target_cet1_EUR_bn_Yr": rwa_target_cet1_yr,
                "RWA_target_total_EUR_bn_Yr": rwa_target_total_yr,
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

        # Audit CET1 allocations (no profit impact)
        for a in alloc_cet1["allocations"]:
            audit_rows.append({
                "Bank": bank,
                "Scenario": row["Scenario"],
                "SRT_Efficiency": row["SRT_Efficiency"],
                "Years": int(row["Years"]),
                "Step": "CET1",
                "RWA_target_base_EUR_bn_Yr": rwa_target_yr_base,
                "RWA_target_redeploy_EUR_bn_Yr": rwa_target_redeploy_yr,
                "RWA_target_cet1_EUR_bn_Yr": rwa_target_cet1_yr,
                "RWA_target_total_EUR_bn_Yr": rwa_target_total_yr,
                "Donor": a.donor,
                "Receiver": a.receiver,
                "Exposure_used_EUR_bn_Yr": a.exposure_used_eur_bn,
                "RWA_reduction_EUR_bn_Yr": a.rwa_reduction_eur_bn,
                "Delta_RWA_pp": a.delta_rwa_pp,
                "Abs_net_spread_bps": a.abs_net_spread_bps,
                "SRT_cost_dec": sc_i,
                "Delta_s_eff_dec": a.delta_s_eff_dec,
                "Profit_contrib_EUR_bn_Yr": 0.0,
                "Status": status,
            })

        addl_profit_yr.append(profit)
        rwa_red_yr_achieved.append(achieved_total)            # total capacity-consuming RWA reduction achieved
        exposure_used_yr.append(expo_used_total)              # total donor exposure used (Step1 + Step2)
        assets_redeploy_used_yr.append(assets_redeploy_used)  # redeployment assets only (Step 1)
        status_list.append(status)

    df["RWA_reduction_achieved_Yr"] = rwa_red_yr_achieved
    df["RWA_target_base_Yr"] = rwa_target_base_yr_list
    df["RWA_target_scaled_Yr"] = rwa_target_scaled_yr_list
    df["RWA_target_scale"] = rwa_target_scale_list
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



# Build bank list (used by bank toggles)
bank_list = sorted(banks["Bank Name"].dropna().astype(str).unique().tolist())

# ---------------- Top controls header (3 columns) ----------------
# Controls previously in the right "sidebar" column are now rendered in a dashboard-style header.
import itertools

# Add subtle vertical separators between the three top-control columns.
# Implementation note: CSS targeting Streamlit's generated DOM can be brittle across versions.
# We therefore insert two narrow "separator" columns between the three control columns.

_SEPARATOR_STYLE = "border-left: 1px solid rgba(49, 51, 63, 0.20); height: 900px; margin: 0 auto;"

def _draw_vsep():
    # Large height ensures the line spans the full height of the controls area.
    st.markdown(f"<div style='{_SEPARATOR_STYLE}'></div>", unsafe_allow_html=True)

class _RoundRobinControls:
    def __init__(self, cols):
        self._cols = cols
        self._it = itertools.cycle(range(len(cols)))

    def _next_col(self):
        return self._cols[next(self._it)]

    def __getattr__(self, name):
        col = self._next_col()
        attr = getattr(col, name)
        if callable(attr):
            def _wrapped(*args, **kwargs):
                return attr(*args, **kwargs)
            return _wrapped
        return attr

_top_controls_container = st.container()
with _top_controls_container:
    # Use 5 columns (3 control columns + 2 thin separator columns). Keep gaps small so
    # the 3 control columns retain enough width for sliders/toggles.
    _tc1, _sep1, _tc2, _sep2, _tc3 = st.columns([1, 0.02, 1, 0.02, 1], gap="small")
    with _sep1:
        _draw_vsep()
    with _sep2:
        _draw_vsep()

top_controls = _RoundRobinControls([_tc1, _tc2, _tc3])
top_controls.header("Global Controls")

with _tc1:
    # Time horizon — moved above Scenario
    years = st.slider(
        "Time horizon (years)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    st.markdown("---")

    st.subheader("Scenario (US advantage in bps)")
    scenario_bps = st.slider(
        "US advantage (bp)",
        min_value=0,
        max_value=400,
        value=168,
        step=5,
    )

    

    # Redeployment / CET1-Split — moved below US advantage in left column
    util = st.slider(
        "Redeployment / CET1-Split (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=1,
        help="Defines how RWAs freed by offloading are used. 100% means 100% of freed RWAs are redeployed into new assets and 0% go to CET1 uplift. 0% means 0% are redeployed and 100% go to CET1 uplift. Intermediate values split proportionally.",
        key="util_slider",
    ) / 100.0
st.markdown("---")


with _tc1:
    st.caption("Select one or more banks:")

    def _safe_bank_key(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", str(name))

    # Defaults: prefer LBBW and Deutsche Bank if present; otherwise fall back to first bank
    preferred_defaults = [b for b in ["LBBW", "Deutsche Bank"] if b in bank_list]
    if not preferred_defaults and bank_list:
        preferred_defaults = [bank_list[0]]

    if "selected_banks" not in st.session_state:
        st.session_state["selected_banks"] = preferred_defaults

    selected_banks = []
    for _b in bank_list:
        _k = f"bank_cb_{_safe_bank_key(_b)}"
        default_on = (_b in st.session_state.get("selected_banks", preferred_defaults))
        val = st.checkbox(_b, key=_k, value=default_on)
        if val:
            selected_banks.append(_b)

    st.session_state["selected_banks"] = selected_banks



# ------------------------------------------------------------
# Consistent legend + colors across charts (by Bank)
# ------------------------------------------------------------
# Keep a stable order for legend items (use the sidebar order)
BANK_ORDER = [b for b in bank_list if b in selected_banks]

# Use the active Plotly template colorway (matches PX defaults)
BANK_COLOR_SEQ = list(getattr(pio.templates[pio.templates.default].layout, "colorway", []))
if not BANK_COLOR_SEQ:
    BANK_COLOR_SEQ = px.colors.qualitative.Plotly

# Deterministic mapping: Bank -> color, in BANK_ORDER
BANK_COLOR_MAP = {b: BANK_COLOR_SEQ[i % len(BANK_COLOR_SEQ)] for i, b in enumerate(BANK_ORDER)}



# Defaults requested: time horizon starts at 5 years

# No Country/Region filtering for now
banks_f = banks.copy()

# Filters (Country/Region) removed as requested

# Bank selection
bank_list = sorted(banks_f["Bank Name"].unique().tolist())

# Defaults requested: preselect only LBBW and Deutsche Bank (if present in filtered list)
preferred_defaults = ["Landesbank Baden-Württemberg", "Deutsche Bank AG"]
default_sel = [b for b in preferred_defaults if b in bank_list]
if not default_sel:
    default_sel = bank_list[: min(8, len(bank_list))]
# Per-bank toggles (multiple banks can be selected)


override_srt_cost_bp = override_tax_rate = None





with _tc2:
    st.subheader("SRT efficiencies")

    srt_eff = st.slider(
        "SRT efficiency",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01,
    
        key="srt_eff_slider",
    )
    st.markdown("---")

    st.subheader("Offload Display")

    assets_method = st.radio(
        "Assets offload method",
        ["Simple (RWA density)", "Transition-based (donor RW)"],
        index=1,
        help=(
            "Simple uses Gross_RWA_Offload converted to assets via bank RWA density. "
            "Transition-based uses the transition engine's achieved RWA reduction per year, "
            "converted to assets via the receiver risk weights and summed across contributions."
        ),
    )

    # Bank-level capacity indicator placeholder (filled later after simulation)
    st.markdown("**Capacity indicator (eligible donor RWAs)**")
    bank_capacity_placeholder = st.container()



    




with _tc3:
    st.subheader("Transition engine controls")

    st.markdown("**SRT-eligible share of donor assets (% of each donor bucket, max 10%)**")
    # Defaults requested: availability sliders start at 5 (max is still 10)
    avail_sme = st.slider("SME term available for SRT (%)", 0, 10, 5, 1, key="avail_sme")
    avail_mid = st.slider("Mid-corp non-IG available for SRT (%)", 0, 10, 5, 1, key="avail_mid")
    avail_em = st.slider("EM corporates available for SRT (%)", 0, 10, 5, 1, key="avail_em")
    avail_cre = st.slider("CRE non-HVCRE available for SRT (%)", 0, 10, 5, 1, key="avail_cre")


donor_availability_pct = {
    "B1_SME_TERM": float(avail_sme),
    "B1_MIDCORP_NONIG": float(avail_mid),
    "B1_EM_CORP": float(avail_em),
    "B1_CRE_NON_HVCRE": float(avail_cre),
}

# Optional per-bank donor split override (not used unless you add controls for it)
donor_split_override = None

# Sidebar toggles removed as requested
require_exact = False
# Target tolerance (%) slider removed; fixed default used instead.
tol_pct = 0.5
show_audit = False

# Placeholder for capacity indicator (filled after model run)



# Offload Display toggles removed as requested (fixed defaults)
metric = "Assets (EUR bn)"
agg = "Total (Horizont)"


top_controls.markdown("---")
top_controls.markdown("---")
# Single SRT efficiency (replaces A/B/C/D sliders)
top_controls.markdown("---")

# Validate selections
if not selected_banks:
    st.error("Please select at least one bank in the sidebar.")
    st.stop()

banks_sel = banks_f[banks_f["Bank Name"].isin(selected_banks)].copy()

# Build single-scenario dict
scenarios = {"US_Advantage": scenario_bps}

effs = [round(float(srt_eff), 4)]

portfolio_df = make_portfolio_row(banks_sel)

# Run model for portfolio
sim_port = simulate_offload(years, portfolio_df, scenarios, effs)
roe_port = compute_roe_delta_transitions_greedy(
    sim_port,
    portfolio_df,
    util=util,
    apply_util_target_scaling=True,
    override_srt_cost_bp=override_srt_cost_bp,
    override_tax_rate=override_tax_rate,
    require_exact_target=require_exact,
    target_tolerance_pct=tol_pct,
    donor_availability_pct_by_donor=donor_availability_pct,
)

# Base (unscaled) allocator run for chart (1) base bars
roe_port_base = compute_roe_delta_transitions_greedy(
    sim_port,
    portfolio_df,
    util=util,
    apply_util_target_scaling=False,
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
    apply_util_target_scaling=True,
    override_srt_cost_bp=override_srt_cost_bp,
    override_tax_rate=override_tax_rate,
    require_exact_target=require_exact,
    target_tolerance_pct=tol_pct,
    donor_availability_pct_by_donor=donor_availability_pct,
)

# Base (unscaled) allocator run for chart (1) base bars
roe_df_base = compute_roe_delta_transitions_greedy(
    sim_df,
    banks_sel,
    util=util,
    apply_util_target_scaling=False,
    override_srt_cost_bp=override_srt_cost_bp,
    override_tax_rate=override_tax_rate,
    require_exact_target=require_exact,
    target_tolerance_pct=tol_pct,
    donor_availability_pct_by_donor=donor_availability_pct,
)
sri_df = compute_sri(sim_df, banks_sel)





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


sim_df = _attach_transition_based_assets(sim_df, roe_df_base)
sim_port = _attach_transition_based_assets(sim_port, roe_port_base)

 
# ============================================================
# Charts
# ============================================================

# ---------------- Top controls (formerly sidebar) ----------------
# Render controls in a true horizontal "header" layout (3 columns).
import itertools

class _RoundRobinControls:
    def __init__(self, cols):
        self._cols = cols
        self._it = itertools.cycle(range(len(cols)))

    def _next_col(self):
        return self._cols[next(self._it)]

    def __getattr__(self, name):
        col = self._next_col()
        attr = getattr(col, name)
        if callable(attr):
            def _wrapped(*args, **kwargs):
                return attr(*args, **kwargs)
            return _wrapped
        return attr

top_controls_container = st.container()
with top_controls_container:
    _c1, _c2, _c3 = st.columns(3, gap="large")

top_controls = _RoundRobinControls([_c1, _c2, _c3])
st.title("Bank-specific Offload Simulation")

# ---- Load bank input data automatically ----
# ---- Main layout: charts left, controls right ----
left_col = st.container()  # charts area (controls moved to top header)


with left_col:
    # Place "1) Offload (Simple)" and "2) Offload Complex (ΔROE)" side by side
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.subheader("1) Offload (Simple)")

        # Fixed offload display settings (sidebar toggles removed)
        yv_simple = "Assets_Offloaded_Transition_EUR_bn_Tot"
        yl_simple = "Assets offloaded (EUR bn, total, transition-based)"

        # --- Redeployment / CET1-Split slider effect on simple offload chart ---
        # We stack an additional bar segment on top of each bank's base bar.
        # Factor requested: base * (util/(1-util)). Example util=75% => factor=3x.
        if util >= 1.0:
            st.warning("Redeployment is set to 100%. The stacked factor becomes undefined (division by zero). Using 99.9% for display.")
        _util_disp = min(float(util), 0.999)
        _factor = _util_disp / (1.0 - _util_disp)  # == (1/(1-util) - 1)

        # Build a stacked-bar figure where each bank is its own offsetgroup so stacks don't mix across banks.
        fig1 = go.Figure()

        # Keep stable ordering for x
        scenarios_order = sim_df["Scenario"].astype(str).dropna().unique().tolist()

        for bank in BANK_ORDER:
            d = sim_df[sim_df["Bank"] == bank]
            if d.empty:
                continue
            # Map scenario -> base value
            base_map = {str(r["Scenario"]): float(r.get(yv_simple, 0.0) or 0.0) for _, r in d.iterrows()}
            base_vals = [base_map.get(str(sc), 0.0) for sc in scenarios_order]
            top_vals = [v * _factor for v in base_vals]

            color = BANK_COLOR_MAP.get(bank)

            # Base (old) bar
            fig1.add_trace(
                go.Bar(
                    name=bank,
                    x=scenarios_order,
                    y=base_vals,
                    marker_color=color,
                    offsetgroup=bank,
                    legendgroup=bank,
                    showlegend=True,
                )
            )

            # Stacked top segment (new)
            fig1.add_trace(
                go.Bar(
                    name=f"{bank} (redeployment)",
                    x=scenarios_order,
                    y=top_vals,
                    marker_color=color,
                    marker_opacity=0.35,
                    offsetgroup=bank,
                    legendgroup=bank,
                    showlegend=False,
                )
            )

        fig1.update_layout(
            barmode="stack",
            title="Required offload (transition-based assets) + redeployment stack",
            legend_title_text="Bank",
            legend_orientation="v",
            legend_yanchor="top",
            legend_y=1,
            legend_xanchor="left",
            legend_x=1.02,
        )
        fig1.update_yaxes(title_text=yl_simple)
        fig1.update_xaxes(title_text="")

        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        st.subheader("2) Offload Complex (ΔROE)")

        fig2 = px.bar(
            roe_df,
            x="Scenario",
            y="ROE_delta_bp",
            color="Bank",
            barmode="group",
            category_orders={"Bank": BANK_ORDER},
            color_discrete_map=BANK_COLOR_MAP,
            labels={"ROE_delta_bp": "ΔROE (bp p.a.)", "Scenario": "", "Bank": "Bank"},
            title="ΔROE (bp p.a.) – Banks (transition-based)"
        )
        fig2.update_layout(
            legend_title_text="Bank",
            legend_orientation="v",
            legend_yanchor="top",
            legend_y=1,
            legend_xanchor="left",
            legend_x=1.02,
        )
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")

    # ---- Donor utilization (how much of eligible donor assets are used) ----
    # Per-year utilization, capped at 100% (availability sliders are per-year caps).
    audit_alloc = roe_df.attrs.get("allocations_audit_df")
    if not isinstance(audit_alloc, pd.DataFrame) or audit_alloc.empty:
        st.info("Donor utilization chart not available (allocation audit data missing).")
    else:
        # Filter to the currently selected scenario/efficiency/horizon to avoid accidental averaging
        _scenario_key = list(scenarios.keys())[0] if isinstance(scenarios, dict) and scenarios else None
        _eff_val = effs[0] if isinstance(effs, list) and effs else None

        alloc_f = audit_alloc.copy()
        if _scenario_key is not None and "Scenario" in alloc_f.columns:
            alloc_f = alloc_f[alloc_f["Scenario"] == _scenario_key]
        if _eff_val is not None and "SRT_Efficiency" in alloc_f.columns:
            # Handle numeric or percentage-string representations (e.g. "75%")
            eff_series = alloc_f["SRT_Efficiency"]
            if eff_series.dtype == object:
                eff_num = (
                    eff_series.astype(str)
                    .str.replace("%", "", regex=False)
                    .str.strip()
                    .astype(float) / 100.0
                )
            else:
                eff_num = eff_series.astype(float)

            alloc_f = alloc_f[np.isclose(eff_num, float(_eff_val))]
        if "Years" in alloc_f.columns:
            alloc_f = alloc_f[alloc_f["Years"] == years]

        # Annual exposure used per bank+donor (EUR bn per year)
        donor_used_bank = (
            alloc_f.groupby(["Bank", "Donor"], as_index=False)
            .agg({"Exposure_used_EUR_bn_Yr": "sum"})
        )

        # Eligible donor exposure stock per bank+donor (EUR bn) after applying donor split + availability cap sliders
        donor_elig = donor_eligible_exposure_long(
            banks_sel,
            donor_availability_pct_by_donor=donor_availability_pct,
            donor_split_override_by_bank=donor_split_override,
        )

        donor_util = donor_used_bank.merge(donor_elig, on=["Bank", "Donor"], how="left")

        # Per-year utilization (% of eligible stock used per year)
        donor_util["Utilization_pct_Yr"] = (
            100.0 * donor_util["Exposure_used_EUR_bn_Yr"] / donor_util["Eligible_Exposure_EUR_bn"]
        )
        donor_util.loc[donor_util["Eligible_Exposure_EUR_bn"] <= 0, "Utilization_pct_Yr"] = 0.0
        donor_util["Utilization_pct_Yr"] = donor_util["Utilization_pct_Yr"].clip(lower=0, upper=100)

        # Sort for stable display
        donor_util = donor_util.sort_values(["Donor", "Bank"])

        st.markdown("### 3) Donor utilization – share of eligible donor assets used")

        # Split utilization into CET1 vs Redeployment steps (stacked; redeployment semi-transparent)
        if "Step" in alloc_f.columns:
            used_by_step = (
                alloc_f.groupby(["Bank", "Donor", "Step"], as_index=False)
                .agg({"Exposure_used_EUR_bn_Yr": "sum"})
            )
        else:
            # Backward compatible: treat everything as CET1 (no redeployment split available)
            used_by_step = donor_used_bank.copy()
            used_by_step["Step"] = "CET1"

        # Pivot to get per-step exposure used
        used_piv = (
            used_by_step.pivot_table(
                index=["Bank", "Donor"],
                columns="Step",
                values="Exposure_used_EUR_bn_Yr",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reset_index()
        )

        # Standardize column names
        redeploy_col = "REDEPLOY" if "REDEPLOY" in used_piv.columns else None
        cet1_col = "CET1" if "CET1" in used_piv.columns else None

        used_piv["Exposure_REDEPLOY"] = used_piv[redeploy_col] if redeploy_col else 0.0
        used_piv["Exposure_CET1"] = used_piv[cet1_col] if cet1_col else 0.0
        used_piv["Exposure_TOTAL"] = used_piv["Exposure_REDEPLOY"] + used_piv["Exposure_CET1"]

        # Eligible donor exposure stock per bank+donor (EUR bn) after applying donor split + availability cap sliders
        donor_elig = donor_eligible_exposure_long(
            banks_sel,
            donor_availability_pct_by_donor=donor_availability_pct,
            donor_split_override_by_bank=donor_split_override,
        )

        donor_util = used_piv.merge(donor_elig, on=["Bank", "Donor"], how="left")
        donor_util["Eligible_Exposure_EUR_bn"] = donor_util["Eligible_Exposure_EUR_bn"].fillna(0.0)

        denom = donor_util["Eligible_Exposure_EUR_bn"].replace(0.0, np.nan)

        donor_util["Util_CET1_pct_Yr"] = 100.0 * donor_util["Exposure_CET1"] / denom
        donor_util["Util_REDEPLOY_pct_Yr"] = 100.0 * donor_util["Exposure_REDEPLOY"] / denom
        donor_util = donor_util.fillna(0.0)

        donor_util["Util_CET1_pct_Yr"] = donor_util["Util_CET1_pct_Yr"].clip(lower=0, upper=100)
        donor_util["Util_REDEPLOY_pct_Yr"] = donor_util["Util_REDEPLOY_pct_Yr"].clip(lower=0, upper=100)

        # Sort for stable display
        donor_util = donor_util.sort_values(["Donor", "Bank"])


        # --- Fill top-control capacity indicators (per selected bank) ---
        try:
            donors_required = ["B1_SME_TERM", "B1_MIDCORP_NONIG", "B1_EM_CORP", "B1_CRE_NON_HVCRE"]
            # Compute TOTAL utilization (Redeploy + CET1) on a per-bank, per-donor basis
            donor_util["Util_TOTAL_pct_Yr"] = (donor_util["Util_CET1_pct_Yr"] + donor_util["Util_REDEPLOY_pct_Yr"]).clip(lower=0, upper=200)

            bank_full = {}
            for bank in banks_sel["Bank Name"].astype(str).tolist():
                d = donor_util[donor_util["Bank"] == bank]
                if d.empty:
                    bank_full[bank] = False
                    continue
                # Ensure all required donors are present; missing donors => not "full"
                full_flags = []
                for donor in donors_required:
                    row = d[d["Donor"].astype(str) == donor]
                    if row.empty:
                        full_flags.append(False)
                    else:
                        # Treat as full if utilization is ~100% or above
                        full_flags.append(float(row["Util_TOTAL_pct_Yr"].iloc[0]) >= 99.9)
                bank_full[bank] = all(full_flags)

            if "bank_capacity_placeholder" in globals():
                
                with bank_capacity_placeholder:
                    # Preserve the order of banks as selected in the UI
                    sel_order = st.session_state.get("selected_banks", [])
                    # Fallback to banks_sel order if session state is unavailable
                    banks_in_order = sel_order if sel_order else banks_sel["Bank Name"].astype(str).tolist()

                    for bank in banks_in_order:
                        if bank_full.get(bank, False):
                            st.error(f"🚨 {bank}: all eligible donor buckets fully utilized")
                        else:
                            st.success(f"✅ {bank}: capacity available")
        except Exception:
            # Never fail the app due to indicator rendering
            pass

        # Build stacked grouped bars: base=CET1 (opaque), top=REDEPLOY (semi-transparent)
        fig_util = go.Figure()

        donors_order = ["B1_SME_TERM", "B1_MIDCORP_NONIG", "B1_EM_CORP", "B1_CRE_NON_HVCRE"]

        for bank in BANK_ORDER:
            d = donor_util[donor_util["Bank"] == bank]
            if d.empty:
                continue

            base_map = {str(r["Donor"]): float(r.get("Util_CET1_pct_Yr", 0.0) or 0.0) for _, r in d.iterrows()}
            top_map = {str(r["Donor"]): float(r.get("Util_REDEPLOY_pct_Yr", 0.0) or 0.0) for _, r in d.iterrows()}

            base_vals = [base_map.get(str(x), 0.0) for x in donors_order]
            top_vals = [top_map.get(str(x), 0.0) for x in donors_order]

            color = BANK_COLOR_MAP.get(bank)

            fig_util.add_trace(
                go.Bar(
                    name=bank,
                    x=donors_order,
                    y=base_vals,
                    marker_color=color,
                    offsetgroup=bank,
                    legendgroup=bank,
                    showlegend=True,
                )
            )

            fig_util.add_trace(
                go.Bar(
                    name=f"{bank} (redeployment)",
                    x=donors_order,
                    y=top_vals,
                    marker_color=color,
                    marker_opacity=0.35,
                    offsetgroup=bank,
                    legendgroup=bank,
                    showlegend=False,
                )
            )

        fig_util.update_layout(
            barmode="stack",
            legend_title_text="Bank",
            legend_orientation="v",
            legend_yanchor="top",
            legend_y=1,
            legend_xanchor="left",
            legend_x=1.02,
        )
        fig_util.update_yaxes(title_text="Utilization (% of eligible per year)", range=[0, 100])
        fig_util.update_xaxes(title_text="Donor bucket")

        st.plotly_chart(fig_util, use_container_width=True)

        # --- Donor bucket weights per selected bank (from CSV) ---
        donor_cols = {
            "SME term (%)": "SME_term",
            "Mid-corp non-IG (%)": "MidCorp_nonIG",
            "EM corporates (%)": "EM_corporates",
            "CRE non-HVCRE (%)": "CRE_non_HVCRE",
        }

        cols_present = [c for c in donor_cols.values() if c in banks_sel.columns]

        if cols_present:
            donor_tbl = (
                banks_sel[["Bank Name"] + cols_present]
                .rename(columns={"Bank Name": "Bank", **{v: k for k, v in donor_cols.items()}})
                .sort_values("Bank")
            )

            # Compute per-bank maximum CET1 uplift (US advantage bps) at util=0 that would fully consume donor capacity
            # (i.e., annual effective RWA reduction equals max achievable using all eligible donor buckets).
            try:
                # Eligible donor exposures per bank+donor (EUR bn) already reflect availability sliders and donor split overrides.
                donor_elig_long = donor_eligible_exposure_long(
                    banks_sel,
                    donor_availability_pct_by_donor=donor_availability_pct,
                    donor_split_override_by_bank=donor_split_override,
                )

                # Build a quick lookup: (bank, donor) -> eligible exposure (EUR bn)
                elig_lookup = {}
                for _, r in donor_elig_long.iterrows():
                    elig_lookup[(str(r["Bank"]), str(r["Donor"]))] = float(r.get("Eligible_Exposure_EUR_bn", 0.0) or 0.0)

                # Per selected bank, compute the max annual effective RWA reduction by allocating an oversized target.
                max_adv_map = {}
                yrs_i = float(years) if "years" in globals() else 1.0
                yrs_i = max(1.0, yrs_i)

                for _, b in banks_sel.iterrows():
                    bank = str(b["Bank Name"])
                    R = float(b["Total RWA (EUR bn)"])
                    cet1_ratio = float(b["CET1 Ratio (%)"]) / 100.0
                    C = cet1_ratio * R

                    donor_expo = {
                        "B1_SME_TERM": elig_lookup.get((bank, "B1_SME_TERM"), 0.0),
                        "B1_MIDCORP_NONIG": elig_lookup.get((bank, "B1_MIDCORP_NONIG"), 0.0),
                        "B1_EM_CORP": elig_lookup.get((bank, "B1_EM_CORP"), 0.0),
                        "B1_CRE_NON_HVCRE": elig_lookup.get((bank, "B1_CRE_NON_HVCRE"), 0.0),
                    }

                    # Use the current SRT efficiency setting (doesn't materially affect RWA-reduction quantity if
                    # the allocator only scales effective reduction by eff; but we pass it for consistency).
                    srt_eff_dec = float(srt_eff) if "srt_eff" in globals() else 0.75

                    alloc_cap = allocate_rwa_reduction_equal_receivers(
                        1e12, donor_expo, srt_eff_dec, 0.0
                    )
                    eff_max_yr = float(alloc_cap.get("total_rwa_reduction_eur_bn", 0.0) or 0.0)

                    eff_max_tot = eff_max_yr * yrs_i

                    # Avoid division by zero / negative RWAs
                    eff_max_tot = min(eff_max_tot, max(R - 1e-9, 0.0))

                    if eff_max_tot <= 0 or (R - eff_max_tot) <= 0:
                        max_adv_map[bank] = np.nan
                        continue

                    target_ratio = C / (R - eff_max_tot)
                    delta = target_ratio - cet1_ratio
                    max_adv_map[bank] = delta * 10000.0

                donor_tbl["Max. Reg. Divergence (bp)"] = donor_tbl["Bank"].map(max_adv_map).round(0).astype("Int64")
            except Exception:
                donor_tbl["Max. Reg. Divergence (bp)"] = np.nan

            st.markdown("**Donor bucket weights per selected bank (from input data)**")
            st.dataframe(
                donor_tbl,
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")
    st.subheader("PORTFOLIO (aggregate across selected banks)")

    # Fixed offload metric for portfolio (same as simple chart)
    yv = yv_simple
    yl = yl_simple

    figP1 = px.bar(
        sim_port,
        x="Scenario",
        y=yv,
        color="SRT_Efficiency",
        barmode="group",
        labels={yv: yl, "Scenario": "", "SRT_Efficiency": "SRT-Efficiency"},
        title="Required Offload – Portfolio"
    )

    figP2 = px.bar(
        roe_port,
        x="Scenario",
        y="ROE_delta_bp",
        color="SRT_Efficiency",
        barmode="group",
        labels={"ROE_delta_bp": "ΔROE (bp p.a.)", "Scenario": "", "SRT_Efficiency": "SRT-Efficiency"},
        title="ΔROE (bp p.a.) – Portfolio (transition-based)"
    )

    pcol1, pcol2 = st.columns(2, gap="large")
    with pcol1:
        st.plotly_chart(figP1, use_container_width=True)
    with pcol2:
        st.plotly_chart(figP2, use_container_width=True)


# ---- XLSX Export ----
xlsx_bytes = to_xlsx_bytes(sim_df, roe_df, sri_df)
st.download_button(
    label="Donwload XLSX",
    data=xlsx_bytes,
    file_name=f"offload_banks_{pd.Timestamp.today().date()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
