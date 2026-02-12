import io
import re
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

# Global constant: total assets column name
TOTAL_ASSETS_COL = "Total Assets (EUR bn)"

# Fallback SRT cost used by the model (input CSV values are ignored; overridden via UI slider).
DEFAULT_SRT_COST_PCT = 0.2  # percent (0.2% = 20 bps)

# Multiplier to increase the donor capacity made available to the allocator (annualized capacity).
ALLOCATOR_CAPACITY_MULTIPLIER = 1.0


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
    "B1_SME_TERM": 1.20,
    "B1_MIDCORP_NONIG": 0.90,
    "B1_EM_CORP": 1.00,
    "B1_CRE_NON_HVCRE": 1.00,
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
    donor_availability_pct_by_donor: Optional[Dict[str, float]] = None,
    receiver_split_by_donor: Optional[Dict[str, Dict[str, float]]] = None,
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
        # NEW: Adjust SRT cost by the donor's SRT-eligible share of donor assets (slider, in %).
        # For each donor->receiver cell, we divide the cost by the eligible share (as a decimal).
        # Example: 5% eligible share -> divide by 0.05 -> higher effective cost for scarce eligibility.
        elig_share_dec = 1.0
        if donor_availability_pct_by_donor and d in donor_availability_pct_by_donor:
            try:
                elig_share_dec = float(donor_availability_pct_by_donor.get(d, 100.0)) / 100.0
            except Exception:
                elig_share_dec = 1.0
        elig_share_dec = max(elig_share_dec, 1e-6)
        srt_cost_bp_adj = srt_cost_bp / elig_share_dec
        # Effective spread term (bps). Note: SRT cost is applied once (no factor 2).
        delta_s_eff_bps = delta_rwa_density * eff * abs_spread_bps - 1.0 * srt_cost_bp_adj
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
    receiver_split_by_donor: Optional[Dict[str, Dict[str, float]]] = None,
    donor_availability_pct_by_donor: Optional[Dict[str, float]] = None,
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
            donor_availability_pct_by_donor=donor_availability_pct_by_donor,
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

        # Split across eligible receivers (user-defined weights if provided; otherwise equal)
        cells = cells_by_donor[d]

        # Build receiver weights map for this donor
        w_map = {}
        if receiver_split_by_donor and isinstance(receiver_split_by_donor, dict):
            w_map = receiver_split_by_donor.get(d, {}) or {}

        # Filter weights to eligible receivers (cell[0] is receiver id) and positive values
        weights = []
        receivers = []
        for cell_obj in cells:
            r = cell_obj.receiver
            w = w_map.get(r, None)
            if w is None:
                continue
            try:
                wv = float(w)
            except Exception:
                continue
            if wv > 0:
                receivers.append(r)
                weights.append(wv)

        if len(weights) == 0:
            # fallback: equal split across eligible receivers
            receivers = [c.receiver for c in cells]
            weights = [1.0] * len(receivers)

        # Normalize
        w_sum = float(sum(weights)) if weights else 0.0
        if w_sum <= 0:
            receivers = [c.receiver for c in cells]
            weights = [1.0] * len(receivers)
            w_sum = float(sum(weights))

        w_norm = {r: (w / w_sum) for r, w in zip(receivers, weights)}

        for cell in cells:
            r = cell.receiver
            expo_used = expo_used_total * float(w_norm.get(r, 0.0))
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

def allocate_until_profit_target(
    profit_target_eur_bn_yr: float,
    donor_exposure_eur_bn: Dict[str, float],
    srt_efficiency: float,
    srt_cost_dec: float,
    tax_dec: float,
    receiver_split_by_donor: Optional[Dict[str, Dict[str, float]]] = None,
    donor_availability_pct_by_donor: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Allocator (Segment B) that allocates donor exposure until a *profit* target is met.

    - Uses the same donor->receiver eligibility logic as `allocate_rwa_reduction_equal_receivers`
      (ΔRWA < 0 and ΔSpread > 0, after SRT-cost adjustment).
    - Uses the user-provided receiver split matrix (per donor) when available; otherwise falls back to equal split.
    - Prioritizes donors by *profit per EUR exposure* (weighted-average across the eligible receivers for that donor
      under the receiver split).

    Returns allocations plus achieved profit / RWA reduction / exposure used.
    """
    profit_target = float(profit_target_eur_bn_yr) if profit_target_eur_bn_yr is not None else 0.0
    if not np.isfinite(profit_target) or profit_target <= 0:
        return {
            "allocations": [],
            "total_profit_eur_bn_yr": 0.0,
            "total_rwa_reduction_eur_bn": 0.0,
            "total_exposure_used_eur_bn": 0.0,
            "total_assets_redeploy_used_eur_bn": 0.0,
            "remaining_profit_target_eur_bn_yr": max(0.0, profit_target),
            "ranked_donors": [],
            "status": "NO_PROFIT_TARGET",
        }

    donors = [d for d in B1_DONORS if donor_exposure_eur_bn.get(d, 0.0) > 0]

    cells_by_donor = _eligible_cells_by_donor(
        donors=donors,
        srt_efficiency=float(srt_efficiency),
        srt_cost_dec=float(srt_cost_dec),
        delta_rwa_pp=DELTA_RWA_PP_B,
        delta_spread_bps=DELTA_SPREAD_BPS_B,
        donor_availability_pct_by_donor=donor_availability_pct_by_donor,
    )

    if not cells_by_donor:
        return {
            "allocations": [],
            "total_profit_eur_bn_yr": 0.0,
            "total_rwa_reduction_eur_bn": 0.0,
            "total_exposure_used_eur_bn": 0.0,
            "total_assets_redeploy_used_eur_bn": 0.0,
            "remaining_profit_target_eur_bn_yr": profit_target,
            "ranked_donors": [],
            "status": "NO_ELIGIBLE_TRANSITIONS",
        }

    eff = float(srt_efficiency)
    tx = float(tax_dec) if np.isfinite(float(tax_dec)) else 0.0
    tx = max(min(tx, 1.0), 0.0)

    donor_rank = []
    donor_receiver_weights: Dict[str, Dict[str, float]] = {}
    donor_profit_per_expo: Dict[str, float] = {}

    # Pre-compute receiver weights + weighted-average profit per EUR exposure for each donor
    for d, cells in cells_by_donor.items():
        w_map = (receiver_split_by_donor or {}).get(d, {}) if isinstance(receiver_split_by_donor, dict) else {}
        receivers, weights = [], []
        for c in cells:
            r = c.receiver
            w = w_map.get(r, None)
            if w is None:
                continue
            try:
                wv = float(w)
            except Exception:
                continue
            if wv > 0:
                receivers.append(r)
                weights.append(wv)

        if len(weights) == 0:
            receivers = [c.receiver for c in cells]
            weights = [1.0] * len(receivers)

        w_sum = float(sum(weights)) if weights else 0.0
        if w_sum <= 0:
            receivers = [c.receiver for c in cells]
            weights = [1.0] * len(receivers)
            w_sum = float(sum(weights))

        w_norm = {r: (w / w_sum) for r, w in zip(receivers, weights)}
        donor_receiver_weights[d] = w_norm

        avg_delta_s_eff = 0.0
        for c in cells:
            avg_delta_s_eff += float(w_norm.get(c.receiver, 0.0)) * float(c.delta_s_eff_dec)

        profit_per_expo = avg_delta_s_eff * (1.0 - tx)
        donor_profit_per_expo[d] = float(profit_per_expo)
        donor_rank.append((d, float(profit_per_expo), len(cells)))

    # IMPORTANT: consume donors in the fixed, user-expected order:
    # SME term -> Mid-corp -> EM corp -> CRE Non-HV-CRE
    remaining_profit = profit_target
    allocs: List[Allocation] = []
    total_profit = 0.0
    total_rwa_red = 0.0
    total_expo = 0.0
    total_assets_redeploy = 0.0

    for d in B1_DONORS:
        if remaining_profit <= 1e-12:
            break
        if d not in cells_by_donor:
            continue

        profit_per_expo = float(donor_profit_per_expo.get(d, 0.0))
        if profit_per_expo <= 0:
            continue

        expo_avail = float(donor_exposure_eur_bn.get(d, 0.0))
        if expo_avail <= 0:
            continue

        expo_needed = remaining_profit / profit_per_expo
        expo_used_total = min(expo_avail, expo_needed)
        if expo_used_total <= 0:
            continue

        donor_rw = float(DONOR_RISK_WEIGHT.get(d, 0.0))
        red_per_eur = donor_rw * eff
        if red_per_eur <= 0:
            continue

        w_norm = donor_receiver_weights.get(d, {})
        cells = cells_by_donor[d]

        for cell in cells:
            r = cell.receiver
            w = float(w_norm.get(r, 0.0))
            if w <= 0:
                continue

            expo_used = expo_used_total * w
            rwa_red = expo_used * red_per_eur
            assets_redeploy = rwa_red / float(cell.receiver_risk_weight)
            contrib = expo_used * float(cell.delta_s_eff_dec) * (1.0 - tx)

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
            total_profit += contrib

        remaining_profit = max(0.0, profit_target - total_profit)

    status = "OK" if remaining_profit <= 1e-9 else "TARGET_NOT_MET"
    return {
        "allocations": allocs,
        "total_profit_eur_bn_yr": total_profit,
        "total_rwa_reduction_eur_bn": total_rwa_red,
        "total_exposure_used_eur_bn": total_expo,
        "total_assets_redeploy_used_eur_bn": total_assets_redeploy,
        "remaining_profit_target_eur_bn_yr": remaining_profit,
        "ranked_donors": [(d, float(ppe), int(n)) for d, ppe, n in donor_rank],
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
            expo_elig = expo_elig * float(ALLOCATOR_CAPACITY_MULTIPLIER)
            rows.append({"Bank": bank, "Donor": donor, "Eligible_Exposure_EUR_bn": expo_elig})
    return pd.DataFrame(rows)



def compute_max_roe_uplift_map(
    sim_df: pd.DataFrame,
    banks_sel: pd.DataFrame,
    override_srt_cost_bp: float | None = None,
    override_tax_rate: float | None = None,
    donor_availability_pct_by_donor: Optional[Dict[str, float]] = None,
    receiver_split_by_donor: Optional[Dict[str, Dict[str, float]]] = None,
    tol_pct: float = 0.5,
) -> Dict[str, float]:
    """Max annual ROE uplift (bp p.a.) per bank when *all* eligible donor capacity is used for redeployment.

    Implementation mirrors the former 'Max. ROE uplift (bp)' column in the table under chart (3),
    but returns a dict so it can be used for visualization (chart 2).
    """
    try:
        if sim_df is None or sim_df.empty or banks_sel is None or banks_sel.empty:
            return {}

        sim_df_max = sim_df.copy()
        # Oversize the RWA reduction targets so the allocator becomes capacity-constrained.
        sim_df_max["Effective_RWA_Reduction_EUR_bn_Tot"] = 1e12
        sim_df_max["Gross_RWA_Offload_EUR_bn_Tot"] = 1e12

        roe_df_maxcap = compute_roe_delta_transitions_greedy(
            sim_df_max,
            banks_sel,
            roe_target_bp=1e6,  # very high target -> exhaust redeployment capacity
            apply_roe_target=True,
            override_srt_cost_bp=override_srt_cost_bp,
            override_tax_rate=override_tax_rate,
            require_exact_target=False,
            target_tolerance_pct=float(tol_pct),
            donor_availability_pct_by_donor=donor_availability_pct_by_donor,
            receiver_split_by_donor=receiver_split_by_donor,
        )

        if not isinstance(roe_df_maxcap, pd.DataFrame) or roe_df_maxcap.empty:
            return {}

        return (
            roe_df_maxcap.groupby("Bank", dropna=False)["ROE_delta_bp"]
            .max()
            .astype(float)
            .to_dict()
        )
    except Exception:
        return {}

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


def simulate_offload(banks_df: pd.DataFrame, scenarios_bps: dict, srt_efficiencies: List[float]) -> pd.DataFrame:
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

                ann_rwa = gross_rwa if np.isfinite(gross_rwa) else np.nan
                gross_ast = rwa_to_assets(gross_rwa, d) if np.isfinite(gross_rwa) else np.nan
                ann_ast = gross_ast if np.isfinite(gross_ast) else np.nan

                rows.append({
                    "Bank": bank,
                    "Country": b.get("Country", ""),
                    "Region": b.get("Region", ""),
                    "Reporting Period": b.get("Reporting Period", ""),
                    "Scenario": sc_name,
                    "US_CET1_Advantage_bps": float(adv_bps),
                    "SRT_Efficiency": f"{round(eff * 100):.0f}%",
                    "Years": 1,

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
    roe_target_bp: float,
    apply_roe_target: bool = True,
    override_srt_cost_bp: float | None = None,
    override_tax_rate: float | None = None,
    donor_split_override_by_bank: Optional[Dict[str, Dict[str, float]]] = None,
    donor_availability_pct_by_donor: Optional[Dict[str, float]] = None,
    receiver_split_by_donor: Optional[Dict[str, Dict[str, float]]] = None,
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
    # NOTE: SRT cost supplied via CSV is intentionally ignored (controlled via UI slider).
    tax_pct = bmap["Effective Tax Rate (%)"].to_dict()
    assets_total = bmap["Total Assets (EUR bn)"].to_dict()

    DEFAULT_SRT_COST = DEFAULT_SRT_COST_PCT  # percent

    df = sim_df.copy()

    # SRT cost and tax series
    # SRT cost: constant fallback (CSV ignored) unless overridden via slider.
    sc = np.full(len(df), float(DEFAULT_SRT_COST) / 100.0, dtype=float)  # decimal
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

        # Steady-state: no time horizon / annualization
    eff_per_year = eff.to_numpy(dtype=float)  # effective RWA reduction target (steady-state)

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
        #   Step 1 (Redeployment): allocate donor exposure until the *ROE / profit target* is met.
        #   Step 2 (CET1 uplift): allocate the RWAs required for CET1 uplift, *after* Step 1 has consumed capacity.
        #
        # Profit / ROE uplift is computed from Step 1 only.
        #
        # Base (annual effective) CET1-uplift requirement:
        rwa_target_cet1_yr = rwa_target_yr_base

        # ROE target (bp) -> annual profit target (EUR bn / yr)
        try:
            roe_target_bp_i = float(roe_target_bp)
        except Exception:
            roe_target_bp_i = 0.0
        if not np.isfinite(roe_target_bp_i):
            roe_target_bp_i = 0.0
        roe_target_bp_i = max(roe_target_bp_i, 0.0)

        cet1_cap_bn = float(row.get("CET1_Capital_EUR_bn", np.nan))
        if not np.isfinite(cet1_cap_bn) or cet1_cap_bn <= 0:
            profit_target_redeploy_yr = 0.0
        else:
            profit_target_redeploy_yr = (roe_target_bp_i / 10000.0) * cet1_cap_bn

        # For reporting: Step 1 does not have an RWA target anymore (it's profit-driven).
        rwa_target_redeploy_yr = 0.0
        rwa_target_total_yr = rwa_target_cet1_yr

        # store targets for reporting (aligned with df rows)
        rwa_target_base_yr_list.append(rwa_target_yr_base)  # CET1 portion (base)
        rwa_target_scaled_yr_list.append(rwa_target_total_yr)  # placeholder; updated after Step 1
        rwa_target_scale_list.append(0.0)  # placeholder; updated after Step 1

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

        # Increase allocator capacity (e.g., to model additional replenishment / more balance-sheet room)
        donor_expo = {k: float(v) * float(ALLOCATOR_CAPACITY_MULTIPLIER) for k, v in donor_expo.items()}

# Parse SRT efficiency string like "75%" -> 0.75
        srt_eff_str = str(row.get("SRT_Efficiency", "")).replace("%", "").strip()
        try:
            srt_eff_i = float(srt_eff_str) / 100.0
        except ValueError:
            srt_eff_i = 0.0

        # Bank-specific (or overridden) SRT cost in decimal
        sc_i_pre = float(sc.iloc[i]) if hasattr(sc, "iloc") else float(sc)

        # ------------------------------
        # Step 1: Redeployment allocation (profit-target driven)
        # ------------------------------
        tx_i = float(tx.iloc[i]) if hasattr(tx, "iloc") else float(tx)
        alloc_redeploy = allocate_until_profit_target(
            profit_target_redeploy_yr,
            donor_expo,
            srt_eff_i,
            sc_i_pre,
            tax_dec=tx_i,
            receiver_split_by_donor=receiver_split_by_donor,
            donor_availability_pct_by_donor=donor_availability_pct_by_donor,
        )

        achieved_redeploy = float(alloc_redeploy["total_rwa_reduction_eur_bn"])
        expo_used_redeploy = float(alloc_redeploy["total_exposure_used_eur_bn"])
        assets_redeploy_used = float(alloc_redeploy["total_assets_redeploy_used_eur_bn"])
        profit_redeploy = float(alloc_redeploy.get("total_profit_eur_bn_yr", 0.0))
        status_redeploy = str(alloc_redeploy["status"])

        # Update reporting targets now that Step 1 (profit-driven) RWA volume is known
        rwa_target_scaled_yr_list[-1] = float(rwa_target_cet1_yr + achieved_redeploy)
        rwa_target_scale_list[-1] = float(achieved_redeploy / rwa_target_cet1_yr) if rwa_target_cet1_yr > 0 else 0.0

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
            rwa_target_cet1_yr, donor_expo_remaining, srt_eff_i, sc_i_pre,
            receiver_split_by_donor=receiver_split_by_donor,
            donor_availability_pct_by_donor=donor_availability_pct_by_donor
        )

        achieved_cet1 = float(alloc_cet1["total_rwa_reduction_eur_bn"])
        expo_used_cet1 = float(alloc_cet1["total_exposure_used_eur_bn"])
        status_cet1 = str(alloc_cet1["status"])

        # Totals (capacity consumption)
        achieved_total = achieved_redeploy + achieved_cet1
        expo_used_total = expo_used_redeploy + expo_used_cet1

        # ------------------------------
        # Target compliance checks
        # ------------------------------
        if rwa_target_cet1_yr > 0:
            cet1_gap_pct = abs(achieved_cet1 - rwa_target_cet1_yr) / rwa_target_cet1_yr * 100.0 if achieved_cet1 > 0 else 100.0
        else:
            cet1_gap_pct = 0.0

        if np.isfinite(cet1_cap_bn) and cet1_cap_bn > 0:
            roe_achieved_bp = (profit_redeploy / cet1_cap_bn) * 10000.0
        else:
            roe_achieved_bp = 0.0

        roe_gap_bp = roe_achieved_bp - roe_target_bp_i

        # Compose a status that reflects both targets
        status = "OK"

        # Step 1 (ROE) feasibility
        if apply_roe_target and roe_target_bp_i > 0:
            if status_redeploy != "OK":
                status = f"ROE_{status_redeploy}"
            else:
                if roe_gap_bp < -1e-6:
                    status = "ROE_TARGET_NOT_MET"

        # Step 2 (CET1) feasibility / tolerance
        if rwa_target_cet1_yr > 0 and status_cet1 != "OK":
            status = f"{status}_CET1_{status_cet1}" if status != "OK" else f"CET1_{status_cet1}"

        if (not require_exact_target) and (rwa_target_cet1_yr > 0) and (cet1_gap_pct > target_tolerance_pct):
            status = f"{status}_CET1_OUTSIDE_TOL({cet1_gap_pct:.2f}%)"

        # ------------------------------
        # Profit calculation (Step 1 ONLY)
        # ------------------------------
        sc_i = float(sc.iloc[i]) if hasattr(sc, "iloc") else float(sc)
        # tx_i already computed above for Step 1.
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
        "RWA_target_base_Yr",
        "RWA_target_scaled_Yr",
        "RWA_target_scale",
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
    # SRT cost values from the CSV are not used by the model; keep a consistent fallback for display.
    df["SRT Cost (%)"] = df["SRT Cost (%)"].fillna(DEFAULT_SRT_COST_PCT)
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
    # Display-only (model uses the SRT cost slider).
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

# Build country list (used by country multi-select)
if 'Country' in banks.columns:
    country_list = sorted(
        banks['Country'].dropna().astype(str).map(str.strip).replace('', pd.NA).dropna().unique().tolist()
    )
else:
    country_list = []

# ---------------- Top controls header (3 columns) ----------------
# Controls previously in the right "sidebar" column are now rendered in a dashboard-style header.
import itertools

# Add subtle vertical separators between the three top-control columns.
# Implementation note: CSS targeting Streamlit's generated DOM can be brittle across versions.
# We therefore insert two narrow "separator" columns between the three control columns.

_SEPARATOR_STYLE = "border-left: 1px solid rgba(49, 51, 63, 0.20); height: 950px; margin: 0 auto;"

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
    st.caption("Steady-state view (no time horizon)")
    years = 1

    st.markdown("---")

    st.subheader("Scenario")
    scenario_bps = st.slider(
        "CET1-uplift target (bp)",
        min_value=0,
        max_value=400,
        value=168,
        step=1,
        help="Annual CET1-uplift target in basis points",
    )

    # Target ROE uplift (bp) — replaces the old Redeployment/CET1 split slider
    roe_target_bp = st.slider(
        "Annual ROE-uplift target (bps)",
        min_value=0,
        max_value=600,
        value=100,
        step=1,
        help="Target annual ROE uplift in basis points (bp). The model first allocates donor capacity to meet this ROE target (via profit-generating redeployment) and then allocates additional capacity to meet the CET1-uplift target.",
        key="roe_target_slider",
    )


with _tc1:
    st.caption("Select one or more countries:")

    # Default: all countries selected (keeps prior behavior: all banks available)
    if "selected_countries" not in st.session_state:
        st.session_state["selected_countries"] = (["Germany"] if "Germany" in country_list else ([country_list[0]] if len(country_list) else []))

    # Track the previous country selection so we can detect changes and sync bank selections.
    if "prev_selected_countries" not in st.session_state:
        st.session_state["prev_selected_countries"] = list(st.session_state.get("selected_countries", []))

    selected_countries = st.multiselect(
        "Countries",
        options=country_list,
        default=st.session_state.get("selected_countries", ["Germany"]),
        key="selected_countries_multiselect",
        help="Filter the bank universe by country",
    )

    selected_countries = list(selected_countries)
    prev_selected_countries = list(st.session_state.get("prev_selected_countries", []))
    countries_changed = set(selected_countries) != set(prev_selected_countries)

    st.session_state["selected_countries"] = selected_countries

    # Apply the country filter to the bank universe
    if selected_countries:
        banks_f = banks[banks["Country"].astype(str).isin(selected_countries)].copy()
    else:
        banks_f = banks.copy()

    bank_list = sorted(banks_f["Bank Name"].dropna().astype(str).unique().tolist())

    # ------------------------------------------------------------
    # Country -> Bank linkage
    # ------------------------------------------------------------
    # If the country selection changed, automatically (de)select banks so that:
    #   - selecting a country selects *all* banks in that country
    #   - deselecting a country deselects *all* banks in that country
    # After the initial sync on country change, users can still fine-tune banks manually.
    if countries_changed:
        if selected_countries:
            auto_banks = list(bank_list)  # all banks in the selected countries
        else:
            auto_banks = []

        st.session_state["selected_banks"] = auto_banks
        # Also set the widget state so the bank multiselect reflects the sync immediately.
        st.session_state["selected_banks_multiselect"] = auto_banks
        st.session_state["prev_selected_countries"] = selected_countries

    st.markdown("---")
    st.caption("Select one or more banks:")

    # Defaults: prefer LBBW and NordLB if present; otherwise fall back to first bank
    preferred_defaults = [b for b in ["LBBW", "NordLB"] if b in bank_list]
    if not preferred_defaults and bank_list:
        preferred_defaults = [bank_list[0]]

    # Use a single multi-select dropdown instead of many checkboxes
    if "selected_banks" not in st.session_state:
        st.session_state["selected_banks"] = preferred_defaults

    # If the country filter removed some previously selected banks, drop them
    prev_banks = st.session_state.get("selected_banks", [])
    prev_banks = [b for b in prev_banks if b in bank_list]
    default_banks = prev_banks if prev_banks else preferred_defaults

    selected_banks = st.multiselect(
        "Banks",
        options=bank_list,
        default=st.session_state.get("selected_banks_multiselect", default_banks),
        key="selected_banks_multiselect",
        help="Select one or more banks",
    )

    # Keep the rest of the app compatible (it expects st.session_state['selected_banks'])
    st.session_state["selected_banks"] = list(selected_banks)


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


override_srt_cost_bp = override_tax_rate = None
receiver_split_by_donor = None


with _tc2:
    st.subheader("SRT Efficiency/ Cost")

    srt_eff = st.slider(
        "SRT efficiency",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01,
        help="Share of RWA relief in a significant risk transfer (SRT) recognized by regulator",
        key="srt_eff_slider",
    )

    # SRT cost is controlled via slider (CSV-supplied SRT Cost (%) is ignored).
    # SRT cost slider (0–40 bps, default 2 bps)
    override_srt_cost_bp = st.slider(
        "SRT cost (bps)",
        min_value=0,
        max_value=40,
        value=20,
        step=1,
        help=(
            "SRT cost in basis points. This slider overrides/ignores any SRT Cost (%) values "
            "supplied via the CSV."
        ),
        key="srt_cost_bps_slider",
    )
    st.markdown("---")
    # Bank-level capacity indicator placeholder (filled later after simulation)
    st.markdown("**Capacity indicator (eligible donor RWAs)**")
    bank_capacity_placeholder = st.container()


    


with _tc3:
    st.subheader("Transition engine controls")

    st.markdown("**SRT-eligible share of donor assets**")
    # Defaults requested: availability sliders start at 5 (max is still 10)
    avail_sme = st.slider("SME term available for SRT (%) — RW 120%", 0, 20, 10, 1, key="avail_sme")
    avail_mid = st.slider("Mid-corp non-IG available for SRT (%) — RW 90%", 0, 20, 10, 1, key="avail_mid")
    avail_em = st.slider("EM corporates available for SRT (%) — RW 100%", 0, 20, 10, 1, key="avail_em")
    avail_cre = st.slider("CRE non-HVCRE available for SRT (%) — RW 140%", 0, 20, 10, 1, key="avail_cre")

    # Donor availability caps (% of each donor bucket, per year)
    donor_availability_pct = {
        "B1_SME_TERM": float(avail_sme),
        "B1_MIDCORP_NONIG": float(avail_mid),
        "B1_EM_CORP": float(avail_em),
        "B1_CRE_NON_HVCRE": float(avail_cre),
    }


    st.markdown("### Receiver split per donor")
    st.caption(
        "Choose how each donor bucket is split across its eligible receiver portfolios. "
        "Each row is normalized to 100% (eligible columns only). Non-eligible donor→receiver cells are disabled."
    )

    def _bucket_label(x: str) -> str:
        s = str(x)
        # Custom display labels for donor buckets (B1)
        if s in {"B1_SME_TERM", "SME_TERM"}:
            return "SME Term"
        if s in {"B1_MIDCORP_NONIG", "MIDCORP_NONIG"}:
            return "Mid-Corp Non-IG"
        if s in {"B1_EM_CORP", "EM_CORP"}:
            return "EM Corp"
        if s in {"B1_CRE_NON_HVCRE", "CRE_NON_HVCRE"}:
            return "Non-HVCRE"
        return s.replace("B1_", "").replace("B2_", "").replace("_", " ").title()

    # Determine eligible receivers per donor under current settings (eligibility depends on SRT eff/cost and Δ matrices)
    _srt_cost_bps_for_elig = float(override_srt_cost_bp) if override_srt_cost_bp is not None else 2.0
    _eligible_map = {}
    for _d in B1_DONORS:
        _cells_map = _eligible_cells_by_donor(
            donors=[_d],
            srt_efficiency=float(srt_eff),
            srt_cost_dec=float(_srt_cost_bps_for_elig) / 10000.0,
            delta_rwa_pp=DELTA_RWA_PP_B,
            delta_spread_bps=DELTA_SPREAD_BPS_B,
            donor_availability_pct_by_donor=donor_availability_pct,
        )
        _cells = _cells_map.get(_d, [])
        _eligible_map[_d] = [c.receiver for c in _cells]

    # Build a pivot-style table: donors as rows, receivers as columns
    _donor_labels = {_d: _bucket_label(_d) for _d in B1_DONORS}
    _recv_labels = {_r: _bucket_label(_r) for _r in B2_RECEIVERS}

    _rows = []
    for _d in B1_DONORS:
        row = {"Donor": _donor_labels[_d], "_donor_id": _d}
        elig = set(_eligible_map.get(_d, []))
        # Default equal split over eligible receivers
        k = max(len(elig), 0)
        for _r in B2_RECEIVERS:
            col = _recv_labels[_r]
            if _r in elig and k > 0:
                row[col] = 100.0 / float(k)
            else:
                row[col] = 0.0
        _rows.append(row)

    _splits_default = pd.DataFrame(_rows)

    # Initialize session state once
    if "receiver_split_pivot" not in st.session_state:
        st.session_state["receiver_split_pivot"] = _splits_default

    # NOTE: Streamlit's st.data_editor does not support per-cell styling.
    # To grey out the exact same (non-eligible) cells as the preview table, we render
    # a small grid of number_inputs and disable non-eligible cells. Disabled inputs
    # are visually greyed out (CSS below), matching the preview behavior.
    st.markdown(
        """
        <style>
        /* (Optional) If any disabled number inputs exist elsewhere, grey them out */
        div[data-testid="stNumberInput"] input:disabled {
            background-color: #eeeeee !important;
            color: #777777 !important;
            -webkit-text-fill-color: #777777 !important;
        }

        /* Residual (computed) cells rendered as HTML so they always refresh */
        .residual-cell {
            background-color: #eeeeee;
            color: #777777;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.5rem;
            padding: 0.38rem 0.5rem;
            text-align: right;
            font-size: 0.9rem;
            line-height: 1.2;
            width: 100%;
            box-sizing: border-box;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Current values (already normalized to 100% across eligible receivers)
    _cur = st.session_state["receiver_split_pivot"].copy()
    _cur_vals_by_donor: Dict[str, Dict[str, float]] = {d: {} for d in B1_DONORS}
    for _, _rr in _cur.iterrows():
        _d = str(_rr.get("_donor_id", ""))
        if _d not in _cur_vals_by_donor:
            continue
        for _r in B2_RECEIVERS:
            _col = _recv_labels[_r]
            try:
                _cur_vals_by_donor[_d][_r] = float(_rr.get(_col, 0.0) or 0.0)
            except Exception:
                _cur_vals_by_donor[_d][_r] = 0.0

    # Header row
    _hcols = st.columns([1.4] + [1.0] * len(B2_RECEIVERS), gap="small")
    _hcols[0].markdown("**Donor**")
    for j, _r in enumerate(B2_RECEIVERS, start=1):
        _hcols[j].markdown(f"**{_recv_labels[_r]}**")

    # Input grid
    _stored_rows = []
    _last_receiver = B2_RECEIVERS[-1]

    for _d in B1_DONORS:
        elig = set(_eligible_map.get(_d, []))
        _rcols = st.columns([1.4] + [1.0] * len(B2_RECEIVERS), gap="small")
        _rcols[0].markdown(_donor_labels[_d])

        row = {"Donor": _donor_labels[_d], "_donor_id": _d}
        _running_sum = 0.0

        for j, _r in enumerate(B2_RECEIVERS, start=1):
            key = f"receiver_split_{_d}_{_r}"
            default_v = float(_cur_vals_by_donor.get(_d, {}).get(_r, 0.0))

            # Non-eligible cells: render as the same grey HTML cell used for the residual.
            # (We avoid disabled widgets here so all greyed-out cells look identical.)
            if _r not in elig:
                st.session_state[key] = 0.0
                _rcols[j].markdown(
                    "<div class='residual-cell'>0</div>",
                    unsafe_allow_html=True,
                )
                v = 0.0

            # Eligible and NOT last column: user-editable.
            elif _r != _last_receiver:
                # Optional UX guard: cap by remaining so residual never goes negative.
                remaining = max(0.0, 100.0 - _running_sum)
                default_v = max(0.0, min(default_v, remaining))

                v = _rcols[j].number_input(
                    label=_recv_labels[_r],
                    min_value=0.0,
                    max_value=float(remaining),
                    step=1.0,
                    value=float(round(default_v, 0)),
                    disabled=False,
                    label_visibility="collapsed",
                    key=key,
                )
                _running_sum += float(v)

            # Eligible and last column: forced residual so the row always sums to 100%.
            else:
                residual = max(0.0, 100.0 - _running_sum)
                # IMPORTANT: A Streamlit widget with a `key` will keep its value from the
                # previous rerun, and the `value=` argument is only used for initialization.
                # To ensure the residual cell *always* updates immediately when upstream
                # inputs change, we render it as HTML (no widget state).
                _rcols[j].markdown(
                    f"<div class='residual-cell'>{float(round(residual, 0)):.0f}</div>",
                    unsafe_allow_html=True,
                )
                v = float(round(residual, 0))

            row[_recv_labels[_r]] = float(v)

        _stored_rows.append(row)

    _stored = pd.DataFrame(_stored_rows)

    # Force non-eligible cells to 0.0 and normalize eligible cells to sum=1 per donor
    receiver_split_by_donor = {}
    _normalized_rows = []
    for _, rr in _stored.iterrows():
        _d = str(rr["_donor_id"])
        elig = set(_eligible_map.get(_d, []))
        vals = {}
        s = 0.0
        # collect values for eligible receivers only
        for _r in B2_RECEIVERS:
            if _r not in elig:
                continue
            col = _recv_labels[_r]
            try:
                v = float(rr[col])
            except Exception:
                v = 0.0
            v = max(v, 0.0)
            vals[_r] = v
            s += v

        if s <= 0:
            # fallback to equal split
            if len(elig) > 0:
                receiver_split_by_donor[_d] = {r: 1.0 / float(len(elig)) for r in elig}
            else:
                receiver_split_by_donor[_d] = {}
        else:
            receiver_split_by_donor[_d] = {r: (v / s) for r, v in vals.items() if v > 0}

        # build normalized row for display (0 in non-eligible cells; eligible shown as % summing to 100)
        norm_row = {"Donor": _donor_labels[_d], "_donor_id": _d}
        for _r in B2_RECEIVERS:
            col = _recv_labels[_r]
            if _r not in elig or len(elig) == 0:
                norm_row[col] = 0.0
            else:
                norm_row[col] = 100.0 * receiver_split_by_donor[_d].get(_r, 0.0)
        _normalized_rows.append(norm_row)

    # Persist normalized (so table stays consistent after reruns)
    st.session_state["receiver_split_pivot"] = pd.DataFrame(_normalized_rows)

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
    st.error("Please select at least one bank in the controls above.")
    st.stop()

banks_sel = banks_f[banks_f["Bank Name"].isin(selected_banks)].copy()

# Build single-scenario dict
scenarios = {"Banks": scenario_bps}

effs = [round(float(srt_eff), 4)]

portfolio_df = make_portfolio_row(banks_sel)

# Run model for portfolio
sim_port = simulate_offload(portfolio_df, scenarios, effs)
roe_port = compute_roe_delta_transitions_greedy(
    sim_port,
    portfolio_df,
    roe_target_bp=roe_target_bp,
    apply_roe_target=True,
    override_srt_cost_bp=override_srt_cost_bp,
    override_tax_rate=override_tax_rate,
    require_exact_target=require_exact,
    target_tolerance_pct=tol_pct,
    donor_availability_pct_by_donor=donor_availability_pct,
                        receiver_split_by_donor=receiver_split_by_donor,
)

# Base (unscaled) allocator run for chart (1) base bars
roe_port_base = compute_roe_delta_transitions_greedy(
    sim_port,
    portfolio_df,
    roe_target_bp=roe_target_bp,
    apply_roe_target=False,
    override_srt_cost_bp=override_srt_cost_bp,
    override_tax_rate=override_tax_rate,
    require_exact_target=require_exact,
    target_tolerance_pct=tol_pct,
    donor_availability_pct_by_donor=donor_availability_pct,
                        receiver_split_by_donor=receiver_split_by_donor,
)
sri_port = compute_sri(sim_port, portfolio_df)

# ---- Run model ----
sim_df = simulate_offload(banks_sel, scenarios, effs)
roe_df = compute_roe_delta_transitions_greedy(
    sim_df,
    banks_sel,
    roe_target_bp=roe_target_bp,
    apply_roe_target=True,
    override_srt_cost_bp=override_srt_cost_bp,
    override_tax_rate=override_tax_rate,
    require_exact_target=require_exact,
    target_tolerance_pct=tol_pct,
    donor_availability_pct_by_donor=donor_availability_pct,
                        receiver_split_by_donor=receiver_split_by_donor,
)

# Base (unscaled) allocator run for chart (1) base bars
roe_df_base = compute_roe_delta_transitions_greedy(
    sim_df,
    banks_sel,
    roe_target_bp=roe_target_bp,
    apply_roe_target=False,
    override_srt_cost_bp=override_srt_cost_bp,
    override_tax_rate=override_tax_rate,
    require_exact_target=require_exact,
    target_tolerance_pct=tol_pct,
    donor_availability_pct_by_donor=donor_availability_pct,
                        receiver_split_by_donor=receiver_split_by_donor,
)
sri_df = compute_sri(sim_df, banks_sel)


# ---- Optional alternative assets-offload measure (transition-based) ----
def _attach_transition_based_assets(sim: pd.DataFrame, roe: pd.DataFrame) -> pd.DataFrame:
    """Attach transition-based asset offload columns to the simple simulation output.

    We compute *step-wise* transition-based assets offloaded using the allocation audit trail:

      - ROE step  ("REDEPLOY")
      - CET1 step ("CET1")

    For each step:
      1) Aggregate annual RWA reduction by donor:
           RWA_d = sum_{allocations in step, row-key, donor d} RWA_reduction_EUR_bn_Yr
      2) Convert donor RWA to net assets using donor RW:
           Assets_net_step = sum_d (RWA_d / Donor_RW_d)
      3) Gross-up by SRT efficiency:
           Assets_gross_step = Assets_net_step / efficiency

    Columns attached (EUR bn):
      - Assets_Offloaded_ROE_Transition_EUR_bn_Yr / _Tot
      - Assets_Offloaded_CET1_Transition_EUR_bn_Yr / _Tot
      - Assets_Offloaded_Transition_EUR_bn_Yr / _Tot   (sum of both steps)

    In addition, we merge back the endogenous scaling outputs from the ROE engine:
      - RWA_target_base_Yr
      - RWA_target_scaled_Yr
      - RWA_target_scale
    """
    sim = sim.copy()

    # Ensure we don't create duplicate columns when merging (avoid _x/_y suffixes)
    sim = sim.drop(columns=[
        "Assets_Offloaded_ROE_Transition_EUR_bn_Yr",
        "Assets_Offloaded_ROE_Transition_EUR_bn_Tot",
        "Assets_Offloaded_CET1_Transition_EUR_bn_Yr",
        "Assets_Offloaded_CET1_Transition_EUR_bn_Tot",
        "Assets_Offloaded_Transition_EUR_bn_Yr",
        "Assets_Offloaded_Transition_EUR_bn_Tot",
        "Assets_Offloaded_Transition_net_EUR_bn_Yr",
        "RWA_target_base_Yr",
        "RWA_target_scaled_Yr",
        "RWA_target_scale",
    ], errors="ignore")

    if roe is None or roe.empty:
        return sim

    audit = roe.attrs.get("allocations_audit_df")
    if not isinstance(audit, pd.DataFrame) or audit.empty:
        # Still merge the scaling outputs if present in roe
        key = ["Bank", "Scenario", "SRT_Efficiency", "Years"]
        back_cols = [c for c in ["RWA_target_base_Yr", "RWA_target_scaled_Yr", "RWA_target_scale"] if c in roe.columns]
        if back_cols:
            sim = sim.merge(roe[key + back_cols], on=key, how="left")
        return sim

    key = ["Bank", "Scenario", "SRT_Efficiency", "Years"]

    # Aggregate by (row-key, step, donor)
    grp = (
        audit.groupby(key + ["Step", "Donor"], dropna=False)["RWA_reduction_EUR_bn_Yr"]
        .sum()
        .reset_index()
    )

    # Convert donor RWA to net assets using donor RW
    grp["Donor_RW"] = grp["Donor"].map(DONOR_RISK_WEIGHT).astype(float)
    grp.loc[grp["Donor_RW"] <= 0, "Donor_RW"] = np.nan
    grp["Assets_net_from_donor"] = grp["RWA_reduction_EUR_bn_Yr"] / grp["Donor_RW"]

    # Sum across donors within each step
    assets_net_step = (
        grp.groupby(key + ["Step"], dropna=False)["Assets_net_from_donor"]
        .sum()
        .reset_index()
        .rename(columns={"Assets_net_from_donor": "Assets_Offloaded_Transition_net_EUR_bn_Yr"})
    )

    # Efficiency gross-up (same per row-key)
    eff_dec = assets_net_step["SRT_Efficiency"].astype(str).str.replace("%", "", regex=False).astype(float) / 100.0
    eff_dec = eff_dec.replace(0.0, np.nan)
    assets_net_step["Assets_Offloaded_Transition_EUR_bn_Yr"] = assets_net_step["Assets_Offloaded_Transition_net_EUR_bn_Yr"] / eff_dec
    assets_net_step["Assets_Offloaded_Transition_EUR_bn_Tot"] = assets_net_step["Assets_Offloaded_Transition_EUR_bn_Yr"]  # steady-state: total == annual

    # Pivot step-wise columns
    piv = assets_net_step.pivot_table(
        index=key,
        columns="Step",
        values=["Assets_Offloaded_Transition_EUR_bn_Yr", "Assets_Offloaded_Transition_EUR_bn_Tot"],
        aggfunc="sum",
        dropna=False,
    )

    # Flatten columns
    piv.columns = [f"{m}_{s}" for (m, s) in piv.columns]
    piv = piv.reset_index()

    # Map to friendly names
    rename_map = {
        "Assets_Offloaded_Transition_EUR_bn_Yr_REDEPLOY": "Assets_Offloaded_ROE_Transition_EUR_bn_Yr",
        "Assets_Offloaded_Transition_EUR_bn_Tot_REDEPLOY": "Assets_Offloaded_ROE_Transition_EUR_bn_Tot",
        "Assets_Offloaded_Transition_EUR_bn_Yr_CET1": "Assets_Offloaded_CET1_Transition_EUR_bn_Yr",
        "Assets_Offloaded_Transition_EUR_bn_Tot_CET1": "Assets_Offloaded_CET1_Transition_EUR_bn_Tot",
    }
    piv = piv.rename(columns=rename_map)

    # Ensure missing step columns exist
    for c in [
        "Assets_Offloaded_ROE_Transition_EUR_bn_Yr",
        "Assets_Offloaded_ROE_Transition_EUR_bn_Tot",
        "Assets_Offloaded_CET1_Transition_EUR_bn_Yr",
        "Assets_Offloaded_CET1_Transition_EUR_bn_Tot",
    ]:
        if c not in piv.columns:
            piv[c] = 0.0

    # Total offload (sum of steps)
    piv["Assets_Offloaded_Transition_EUR_bn_Yr"] = piv["Assets_Offloaded_ROE_Transition_EUR_bn_Yr"] + piv["Assets_Offloaded_CET1_Transition_EUR_bn_Yr"]
    piv["Assets_Offloaded_Transition_EUR_bn_Tot"] = piv["Assets_Offloaded_ROE_Transition_EUR_bn_Tot"] + piv["Assets_Offloaded_CET1_Transition_EUR_bn_Tot"]

    # Merge scaling outputs from roe table
    back_cols = [c for c in ["RWA_target_base_Yr", "RWA_target_scaled_Yr", "RWA_target_scale"] if c in roe.columns]
    roe_back = roe[key + back_cols].drop_duplicates() if back_cols else roe[key].drop_duplicates()

    sim = sim.merge(piv[key + [
        "Assets_Offloaded_ROE_Transition_EUR_bn_Yr",
        "Assets_Offloaded_ROE_Transition_EUR_bn_Tot",
        "Assets_Offloaded_CET1_Transition_EUR_bn_Yr",
        "Assets_Offloaded_CET1_Transition_EUR_bn_Tot",
        "Assets_Offloaded_Transition_EUR_bn_Yr",
        "Assets_Offloaded_Transition_EUR_bn_Tot",
    ]], on=key, how="left")

    sim = sim.merge(roe_back, on=key, how="left")

    # Fill NaNs produced by missing steps/keys
    for c in [
        "Assets_Offloaded_ROE_Transition_EUR_bn_Yr",
        "Assets_Offloaded_ROE_Transition_EUR_bn_Tot",
        "Assets_Offloaded_CET1_Transition_EUR_bn_Yr",
        "Assets_Offloaded_CET1_Transition_EUR_bn_Tot",
        "Assets_Offloaded_Transition_EUR_bn_Yr",
        "Assets_Offloaded_Transition_EUR_bn_Tot",
    ]:
        sim[c] = sim[c].fillna(0.0)

    return sim


# Attach step-wise transition-based offload columns for Chart 1/3 (CET1 vs ROE split)
sim_df = _attach_transition_based_assets(sim_df, roe_df)

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
    top_left, top_right = st.columns(2, gap="large")

    with top_left:
        st.subheader("1) CET1-Uplift (end-state)")

        # CET1 uplift is a level (stock) change. It does not "cumulate" the way ROE does.
        # Filled bars show the end-state CET1 uplift target (bp).
        # Light-grey outline shows remaining headroom up to the max-capacity cap (Max. Reg. Divergence, bp).

        def _compute_cet1_target_df():
            base_cols = [c for c in ["Scenario", "SRT_Efficiency", "Bank"] if c in sim_df.columns]
            base = sim_df[base_cols].drop_duplicates().copy()
            try:
                tgt = float(scenario_bps)
            except Exception:
                tgt = 0.0
            base["CET1_target_bp"] = tgt
            return base[["Scenario", "SRT_Efficiency", "Bank", "CET1_target_bp"]]

        def _compute_max_reg_divergence_map_endstate():
            try:
                donor_elig_long = donor_eligible_exposure_long(
                    banks_sel,
                    donor_availability_pct_by_donor=donor_availability_pct,
                    donor_split_override_by_bank=donor_split_override,
                )
                elig_lookup = {(str(r["Bank"]), str(r["Donor"])): float(r.get("Eligible_Exposure_EUR_bn", 0.0) or 0.0) for _, r in donor_elig_long.iterrows()}
            except Exception:
                elig_lookup = {}

            cap_map = {}
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

                srt_eff_dec = float(srt_eff) if "srt_eff" in globals() else 0.75

                try:
                    alloc_cap = allocate_rwa_reduction_equal_receivers(
                        1e12, donor_expo, srt_eff_dec, 0.0,
                        donor_availability_pct_by_donor=donor_availability_pct,
                    )
                    eff_max_yr = float(alloc_cap.get("total_rwa_reduction_eur_bn", 0.0) or 0.0)
                except Exception:
                    eff_max_yr = 0.0

                eff_max_tot = float(eff_max_yr)  # donor stock (no x years)
                eff_rwa_tot = min(float(eff_max_tot), max(R - 1e-9, 0.0))

                if eff_rwa_tot <= 0 or (R - eff_rwa_tot) <= 0:
                    cap_bp = np.nan
                else:
                    target_ratio = C / (R - eff_rwa_tot)
                    cap_bp = (target_ratio - cet1_ratio) * 10000.0

                cap_map[str(bank).strip()] = cap_bp

            return cap_map

        def _plot_cet1_endstate():
            df = _compute_cet1_target_df()
            if df.empty:
                return go.Figure().update_layout(title="CET1 uplift target – Banks (transition-based)")

            scenarios_order = df["Scenario"].astype(str).dropna().unique().tolist()
            cap_map = {str(k).strip(): v for k, v in (_compute_max_reg_divergence_map_endstate() or {}).items()}
            banks_present = df["Bank"].astype(str).dropna().unique().tolist()
            banks_in_df = [b for b in BANK_ORDER if b in banks_present] if "BANK_ORDER" in globals() else banks_present

            fig = go.Figure()
            for bank in banks_in_df:
                d = df[df["Bank"].astype(str).str.strip() == str(bank).strip()]
                if d.empty:
                    continue

                cur_map = {str(r["Scenario"]): float(r.get("CET1_target_bp", 0.0) or 0.0) for _, r in d.iterrows()}
                cur_vals = [cur_map.get(str(sc), 0.0) for sc in scenarios_order]

                max_val = float(cap_map.get(str(bank).strip(), np.nan))
                pot_vals = [max(max_val - float(v), 0.0) for v in cur_vals] if np.isfinite(max_val) else [0.0 for _ in cur_vals]

                color = BANK_COLOR_MAP.get(bank)

                fig.add_trace(
                    go.Bar(
                        name=bank,
                        x=scenarios_order,
                        y=cur_vals,
                        marker_color=color,
                        offsetgroup=bank,
                        legendgroup=bank,
                        showlegend=True,
                    )
                )

                fig.add_trace(
                    go.Bar(
                        name=f"{bank} (potential)",
                        x=scenarios_order,
                        y=pot_vals,
                        base=cur_vals,
                        marker=dict(
                            color="rgba(0,0,0,0)",
                            line=dict(color="lightgrey", width=2),
                        ),
                        offsetgroup=bank,
                        legendgroup=bank,
                        showlegend=False,
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color="lightgrey", width=2),
                    name="CET1 potential (max-capacity)",
                    showlegend=True,
                )
            )

            fig.update_layout(
                barmode="group",
                title="CET1 uplift target (bp) – Banks (transition-based)",
                legend_title_text="Bank",
                legend_orientation="v",
                legend_yanchor="top",
                legend_y=1,
                legend_xanchor="left",
                legend_x=1.02,
            )
            fig.update_yaxes(title_text="ΔCET1 ratio (bp, end-state)")
            fig.update_xaxes(title_text="")
            return fig

        st.plotly_chart(_plot_cet1_endstate(), use_container_width=True)
    with top_right:
        st.subheader("2) ROE-Uplift")

        # Steady-state: only the annual view is relevant.
        roe_df_view = roe_df.copy()

        # Portfolio average reference line uses a CET1-capital proxy weight
        _w = banks_sel[["Bank Name", "Total RWA (EUR bn)", "CET1 Ratio (%)"]].copy()
        _w["weight"] = _w["Total RWA (EUR bn)"] * _w["CET1 Ratio (%)"]

        def _plot_roe(y_col: str, y_label: str, title: str):
            _tmp = roe_df_view.merge(_w[["Bank Name", "weight"]], left_on="Bank", right_on="Bank Name", how="left")
            _tmp = _tmp.dropna(subset=["weight"])

            roe_port_avg = (
                _tmp.groupby(["Scenario", "SRT_Efficiency"], as_index=False)
                    .apply(lambda g: pd.Series({y_col: float(np.average(g[y_col], weights=g["weight"]))}))
            )

            # --- Max-capacity ROE potential (outline) ---
            max_roe_map = compute_max_roe_uplift_map(
                sim_df=sim_df,
                banks_sel=banks_sel,
                override_srt_cost_bp=override_srt_cost_bp,
                override_tax_rate=override_tax_rate,
                donor_availability_pct_by_donor=donor_availability_pct,
                receiver_split_by_donor=receiver_split_by_donor,
                tol_pct=tol_pct,
            )

            scenarios_order = roe_df_view["Scenario"].astype(str).dropna().unique().tolist()

            fig = go.Figure()

            for bank in BANK_ORDER:
                d = roe_df_view[roe_df_view["Bank"] == bank]
                if d.empty:
                    continue

                cur_map = {str(r["Scenario"]): float(r.get(y_col, 0.0) or 0.0) for _, r in d.iterrows()}
                cur_vals = [cur_map.get(str(sc), 0.0) for sc in scenarios_order]

                max_ann = float(max_roe_map.get(bank, np.nan))
                if np.isfinite(max_ann):
                    max_val = max_ann
                    pot_vals = [max(max_val - float(v), 0.0) for v in cur_vals]
                else:
                    pot_vals = [0.0 for _ in cur_vals]

                color = BANK_COLOR_MAP.get(bank)

                # Filled bar = achieved ΔROE
                fig.add_trace(
                    go.Bar(
                        name=bank,
                        x=scenarios_order,
                        y=cur_vals,
                        marker_color=color,
                        offsetgroup=bank,
                        legendgroup=bank,
                        showlegend=True,
                    )
                )

                # Outline bar (stacked on top of the achieved value) = remaining potential up to max-capacity
                fig.add_trace(
                    go.Bar(
                        name=f"{bank} (potential)",
                        x=scenarios_order,
                        y=pot_vals,
                        base=cur_vals,
                        marker=dict(
                            color="rgba(0,0,0,0)",
                            line=dict(color='lightgrey', width=2),
                        ),
                        offsetgroup=bank,
                        legendgroup=bank,
                        showlegend=False,
                        hovertemplate=(
                            "Scenario=%{x}<br>"
                            + "Potential to max=%{y:.1f}<br>"
                            + "<extra></extra>"
                        ),
                    )
                )

            # Single legend entry explaining the outline
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color='lightgrey', width=2),
                    name="ROE potential (max-capacity)",
                    showlegend=True,
                )
            )

            fig.update_layout(
                barmode="group",
                title=title,
                legend_title_text="Bank",
                legend_orientation="v",
                legend_yanchor="top",
                legend_y=1,
                legend_xanchor="left",
                legend_x=1.02,
            )
            fig.update_yaxes(title_text=y_label)
            fig.update_xaxes(title_text="")
            return fig

        fig2 = _plot_roe(
            y_col="ROE_delta_bp",
            y_label="ΔROE (bp p.a.)",
            title="ΔROE (bp p.a.) – Banks (transition-based)",
        )
        st.plotly_chart(fig2, use_container_width=True)
    bottom_left, bottom_right = st.columns(2, gap="large")

    with bottom_left:
        st.subheader("3) Total Asset Offload")

        # Steady-state: only the annual view is relevant.

        def _build_offload_fig(yv_cet1: str, yv_roe: str, y_label: str, title: str):
            # Solid bars   = assets offloaded to achieve the CET1-uplift (Step "CET1")
            # Transparent  = additional assets offloaded to achieve the ROE-uplift (Step "REDEPLOY")
            fig = go.Figure()
            scenarios_order = sim_df["Scenario"].astype(str).dropna().unique().tolist()

            for bank in BANK_ORDER:
                d = sim_df[sim_df["Bank"] == bank]
                if d.empty:
                    continue

                cet1_map = {str(r["Scenario"]): float(r.get(yv_cet1, 0.0) or 0.0) for _, r in d.iterrows()}
                roe_map  = {str(r["Scenario"]): float(r.get(yv_roe, 0.0) or 0.0)  for _, r in d.iterrows()}

                base_vals = [cet1_map.get(str(sc), 0.0) for sc in scenarios_order]
                top_vals  = [roe_map.get(str(sc), 0.0)  for sc in scenarios_order]

                color = BANK_COLOR_MAP.get(bank)

                fig.add_trace(
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
                fig.add_trace(
                    go.Bar(
                        name=bank,
                        x=scenarios_order,
                        y=top_vals,
                        marker_color=color,
                        opacity=0.35,
                        offsetgroup=bank,
                        legendgroup=bank,
                        showlegend=False,
                    )
                )

            fig.update_layout(
                barmode="stack",
                title=title,
                legend_title_text="Bank",
                legend_orientation="v",
                legend_yanchor="top",
                legend_y=1,
                legend_xanchor="left",
                legend_x=1.02,
            )
            fig.update_yaxes(title_text=y_label)
            fig.update_xaxes(title_text="")
            return fig

        fig3 = _build_offload_fig(
            yv_cet1="Assets_Offloaded_CET1_Transition_EUR_bn_Yr",
            yv_roe="Assets_Offloaded_ROE_Transition_EUR_bn_Yr",
            y_label="Assets offloaded (EUR bn p.a., transition-based)",
            title="Required offload (transition-based assets): CET1 + ROE split (annual)",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with bottom_right:
        st.markdown("### 4) Donor utilization – share of eligible donor assets used")
        st.markdown(
            "<div style='color: #666666; font-size: 0.9rem; margin-top: -6px; margin-bottom: 12px;'>"
            "Donor buckets are prioritized according to best transition efficiency = receiver–donor net spread / receiver-donor risk weight improvement. "
            "Transition efficiency descending from left to right."
            "</div>",
            unsafe_allow_html=True,
        )

        # Split utilization into CET1 vs Redeployment steps (stacked; redeployment semi-transparent)

        # Allocations audit (from ROE transition engine) used to split utilization by step
        alloc_f = None
        try:
            alloc_f = getattr(roe_df, "attrs", {}).get("allocations_audit_df", None)
        except Exception:
            alloc_f = None
        if not isinstance(alloc_f, pd.DataFrame):
            alloc_f = pd.DataFrame()
        # Keep only combinations that are currently in sim_df (defensive against stale audits)
        try:
            key_cols = ["Scenario", "SRT_Efficiency", "Bank", "Years"]
            if all(c in alloc_f.columns for c in key_cols) and all(c in sim_df.columns for c in key_cols):
                _keys = sim_df[key_cols].drop_duplicates()
                alloc_f = alloc_f.merge(_keys, on=key_cols, how="inner")
        except Exception:
            pass

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

        # --- STOCK CONSISTENCY ---
        # With donor assets treated as a *stock* (distributed across the horizon inside the annual transition engine),
        # Exposure_CET1 / Exposure_REDEPLOY are per-year usage. For utilization and the capacity indicator we want
        # TOTAL usage over the horizon vs the eligible STOCK pool.
        denom = donor_util["Eligible_Exposure_EUR_bn"].replace(0.0, np.nan)

        donor_util["Util_CET1_pct"] = 100.0 * (donor_util["Exposure_CET1"]) / denom
        donor_util["Util_REDEPLOY_pct"] = 100.0 * (donor_util["Exposure_REDEPLOY"]) / denom
        donor_util = donor_util.fillna(0.0)

        donor_util["Util_CET1_pct"] = donor_util["Util_CET1_pct"].clip(lower=0, upper=100)
        donor_util["Util_REDEPLOY_pct"] = donor_util["Util_REDEPLOY_pct"].clip(lower=0, upper=100)

        # Sort for stable display
        donor_util = donor_util.sort_values(["Donor", "Bank"])


        # --- Fill top-control capacity indicators (per selected bank) ---
        try:
            donors_required = ["B1_SME_TERM", "B1_MIDCORP_NONIG", "B1_EM_CORP", "B1_CRE_NON_HVCRE"]
            # Compute TOTAL utilization (Redeploy + CET1) on a per-bank, per-donor basis
            donor_util["Util_TOTAL_pct"] = (donor_util["Util_CET1_pct"] + donor_util["Util_REDEPLOY_pct"]).clip(lower=0, upper=200)

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
                        full_flags.append(float(row["Util_TOTAL_pct"].iloc[0]) >= 99.9)
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

            base_map = {str(r["Donor"]): float(r.get("Util_CET1_pct", 0.0) or 0.0) for _, r in d.iterrows()}
            top_map = {str(r["Donor"]): float(r.get("Util_REDEPLOY_pct", 0.0) or 0.0) for _, r in d.iterrows()}

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
                        1e12, donor_expo, srt_eff_dec, 0.0,
                        donor_availability_pct_by_donor=donor_availability_pct,
                    )
                    eff_max_yr = float(alloc_cap.get("total_rwa_reduction_eur_bn", 0.0) or 0.0)

                    eff_max_tot = eff_max_yr  # donor stock (no x years)

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


            # --- RWA offload (CET1 uplift + ROE uplift) per bank ---
            try:
                audit_df = getattr(roe_df, "attrs", {}).get("allocations_audit_df", None)
                if isinstance(audit_df, pd.DataFrame) and not audit_df.empty:
                    # Sum annual RWA reductions by step and convert to total-horizon RWAs
                    by_step = (
                        audit_df.groupby(["Bank", "Step"], dropna=False)["RWA_reduction_EUR_bn_Yr"]
                        .sum()
                        .unstack(fill_value=0.0)
                    )
                    rwa_sum_map = (by_step.get("CET1", 0.0) + by_step.get("REDEPLOY", 0.0)).to_dict()
                    donor_tbl["RWA freed (EUR bn)"] = donor_tbl["Bank"].map(rwa_sum_map).round(0).astype("Int64").astype("Int64")
                else:
                    donor_tbl["RWA freed (EUR bn)"] = np.nan
            except Exception:
                donor_tbl["RWA freed (EUR bn)"] = np.nan

            st.markdown("**Donor bucket weights per selected bank (from input data)**")
            donor_tbl_display = donor_tbl.drop(columns=["Max. Reg. Divergence (bp)"], errors="ignore")

            st.dataframe(
                donor_tbl_display,
                column_config={
                    "RWA freed (EUR bn)": st.column_config.NumberColumn(
                        "RWA freed (EUR bn)", format="%.0f"
                    )
                },
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")
    st.subheader("PORTFOLIO (aggregate across selected banks)")

    # Required offload portfolio = sum of chart (1) across banks (keep CET1/ROE split)
    yl = "Offloaded assets (EUR bn, total, transition-based) — sum across banks"

    _tmp_port = sim_df[[        "Scenario",
        "SRT_Efficiency",
        "Bank",
        "Assets_Offloaded_CET1_Transition_EUR_bn_Tot",
        "Assets_Offloaded_ROE_Transition_EUR_bn_Tot",
    ]].copy()
    _tmp_port["Assets_Offloaded_CET1_Transition_EUR_bn_Tot"] = _tmp_port["Assets_Offloaded_CET1_Transition_EUR_bn_Tot"].fillna(0.0)
    _tmp_port["Assets_Offloaded_ROE_Transition_EUR_bn_Tot"] = _tmp_port["Assets_Offloaded_ROE_Transition_EUR_bn_Tot"].fillna(0.0)

    port_base = (
        _tmp_port.groupby(["Scenario", "SRT_Efficiency"], as_index=False)[
            ["Assets_Offloaded_CET1_Transition_EUR_bn_Tot", "Assets_Offloaded_ROE_Transition_EUR_bn_Tot"]
        ].sum()
    )

    port_long = port_base.melt(
        id_vars=["Scenario", "SRT_Efficiency"],
        value_vars=["Assets_Offloaded_CET1_Transition_EUR_bn_Tot", "Assets_Offloaded_ROE_Transition_EUR_bn_Tot"],
        var_name="Component",
        value_name="Assets_offloaded_EUR_bn",
    )
    port_long["Component"] = port_long["Component"].map({
        "Assets_Offloaded_CET1_Transition_EUR_bn_Tot": "CET1-uplift",
        "Assets_Offloaded_ROE_Transition_EUR_bn_Tot": "ROE-uplift",
    })
    # Ensure one bar per Scenario/Component/(SRT_Efficiency) so Plotly can stack reliably
    port_long = (
        port_long.groupby(["Scenario", "SRT_Efficiency", "Component"], as_index=False)["Assets_offloaded_EUR_bn"]
        .sum()
    )


    figP1 = px.bar(
        port_long,
        x="Scenario",
        y="Assets_offloaded_EUR_bn",
        color="Component",
        barmode="stack",
        category_orders={"Component": ["CET1-uplift", "ROE-uplift"]},
        facet_col="SRT_Efficiency" if len(effs) > 1 else None,
        labels={
            "Assets_offloaded_EUR_bn": yl,
            "Scenario": "",
            "SRT_Efficiency": "SRT-Efficiency",
            "Component": "",
        },
        title="Required Offload – Portfolio (sum across selected banks)",
    )

    # Force true stacking (Plotly stacks only within the same offsetgroup; px.bar may set different offsetgroups)
    for _tr in figP1.data:
        _tr.offsetgroup = "portfolio"
        _tr.alignmentgroup = "portfolio"
    figP1.update_layout(barmode="stack")

    # Portfolio ΔROE: weighted average of bank-level ΔROE (from chart 2),
    # weights = Total RWA (EUR bn) × CET1 Ratio (%), taken from input data.
    _w = banks_sel[["Bank Name", "Total RWA (EUR bn)", "CET1 Ratio (%)"]].copy()
    _w["weight"] = _w["Total RWA (EUR bn)"] * _w["CET1 Ratio (%)"]

    _tmp = roe_df.merge(_w[["Bank Name", "weight"]], left_on="Bank", right_on="Bank Name", how="left")
    _tmp = _tmp.dropna(subset=["weight"])

    roe_port = (
        _tmp.groupby(["Scenario", "SRT_Efficiency"], as_index=False)
            .apply(lambda g: pd.Series({"ROE_delta_bp": float(np.average(g["ROE_delta_bp"], weights=g["weight"]))}))
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
    figP1.update_layout(legend_traceorder="reversed")

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
