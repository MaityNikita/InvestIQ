"""
=============================================================================
  recommendation_engine.py — Personalised Financial Advice Engine
  AI Financial Advisory Chatbot — InvestIQ
=============================================================================
  Generates structured recommendations across:
    - Risk profiling (Conservative / Moderate / Aggressive)
    - Portfolio allocation suggestions
    - Budgeting strategies
    - Investment product recommendations
    - Savings growth projections
    - Tax-efficient strategies (India context)
=============================================================================
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Literal

# ── Types ─────────────────────────────────────────────────────────────────────
RiskTier = Literal["conservative", "moderate", "aggressive"]


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserProfile:
    age:            int
    monthly_income: float       # INR
    monthly_expenses: float     # INR
    current_savings: float      # INR
    investment_horizon: int     # years
    risk_answers: list[int]     # 3 quiz answers (1=low, 2=med, 3=high)
    existing_investments: dict  = field(default_factory=dict)

    @property
    def risk_tier(self) -> RiskTier:
        score = sum(self.risk_answers)
        if score <= 5:
            return "conservative"
        elif score <= 8:
            return "moderate"
        else:
            return "aggressive"

    @property
    def monthly_surplus(self) -> float:
        return max(self.monthly_income - self.monthly_expenses, 0)

    @property
    def savings_rate(self) -> float:
        if self.monthly_income == 0:
            return 0.0
        return self.monthly_surplus / self.monthly_income


@dataclass
class Recommendation:
    risk_tier:        RiskTier
    portfolio_mix:    dict          # {"Equity": 60, "Debt": 30, "Gold": 10}
    monthly_invest:   float         # recommended monthly investment in INR
    products:         list[str]     # e.g. ["Nifty 50 Index Fund", "PPF"]
    budgeting_tips:   list[str]
    savings_target:   float         # 6-month emergency fund target
    projected_corpus: float         # future value after horizon
    advice_text:      str


# ═══════════════════════════════════════════════════════════════════════════════
#  ALLOCATION TABLES
# ═══════════════════════════════════════════════════════════════════════════════

_ALLOCATION = {
    "conservative": {"Equity": 25, "Debt": 55, "Gold": 10, "Cash/FD": 10},
    "moderate":     {"Equity": 55, "Debt": 30, "Gold":  8, "Cash/FD":  7},
    "aggressive":   {"Equity": 75, "Debt": 15, "Gold":  5, "Cash/FD":  5},
}

_PRODUCTS = {
    "conservative": [
        "Liquid Mutual Funds (emergency corpus)",
        "Public Provident Fund (PPF) — 7.1% tax-free",
        "Government Bonds / G-Sec",
        "Bank Fixed Deposits (5-year tax-saver)",
        "Conservative Hybrid Mutual Funds",
        "National Savings Certificate (NSC)",
    ],
    "moderate": [
        "Nifty 50 Index Fund (large-cap exposure)",
        "Flexi-Cap Mutual Funds (SIP)",
        "PPF (debt anchor)",
        "Corporate Bond Funds (AAA-rated)",
        "Sovereign Gold Bonds (SGBs)",
        "ELSS Funds (tax saving under 80C)",
    ],
    "aggressive": [
        "Mid-cap & Small-cap Index Funds",
        "Sectoral / Thematic Funds (IT, Pharma)",
        "Direct Equity — NSE IT basket (TCS, Infosys, HCL)",
        "US Tech ETFs (diversification)",
        "REITs (real-estate exposure)",
        "NPS Tier-I equity allocation (E scheme)",
    ],
}

_EXPECTED_RETURNS = {
    "conservative": 0.065,   # ~6.5% p.a.
    "moderate":     0.10,    # ~10% p.a.
    "aggressive":   0.14,    # ~14% p.a.
}

_BUDGETING_TIPS = {
    "conservative": [
        "Follow the 50-30-20 rule: 50% Needs, 30% Wants, 20% Savings.",
        "Build a 6-month emergency fund before investing.",
        "Automate recurring deposits to avoid impulse spending.",
        "Track all expenses weekly using a free app (e.g., Walnut, Money View).",
        "Avoid high-interest personal loans; pay off debt before investing.",
    ],
    "moderate": [
        "Target a 25–30% savings rate on gross income.",
        "Use SIPs to invest on auto-pilot every month.",
        "Review and rebalance portfolio every 6 months.",
        "Keep 3- to 6-month emergency fund in a liquid fund.",
        "Maximise Section 80C (₹1.5L) and 80D (health insurance) deductions.",
    ],
    "aggressive": [
        "Invest 35%+ of income to maximise compounding.",
        "Use a core-satellite approach: 70% index + 30% tactical bets.",
        "Rebalance annually; trim winners and add to laggards.",
        "Consider international diversification (US, Asia) for currency hedge.",
        "Keep 3 months of expenses liquid, rest deployed in market.",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_recommendation(profile: UserProfile) -> Recommendation:
    """
    Generate a complete personalised financial recommendation for a user.
    """
    tier = profile.risk_tier

    # Monthly investable amount (cap at 40% of income for safety)
    invest_frac   = {"conservative": 0.15, "moderate": 0.25, "aggressive": 0.35}
    monthly_invest = min(
        profile.monthly_surplus * invest_frac[tier],
        profile.monthly_income  * 0.40,
    )

    # Future value: FV = PV*(1+r)^n + PMT*[((1+r)^n - 1)/r]
    r   = _EXPECTED_RETURNS[tier]
    n   = profile.investment_horizon
    pv  = profile.current_savings
    pmt = monthly_invest * 12   # annual contribution
    fv  = pv * (1 + r)**n + pmt * (((1 + r)**n - 1) / r)

    savings_target = profile.monthly_expenses * 6   # 6-month emergency fund

    advice = _build_advice_text(profile, tier, monthly_invest, fv)

    return Recommendation(
        risk_tier        = tier,
        portfolio_mix    = _ALLOCATION[tier],
        monthly_invest   = round(monthly_invest, 2),
        products         = _PRODUCTS[tier],
        budgeting_tips   = _BUDGETING_TIPS[tier],
        savings_target   = round(savings_target, 2),
        projected_corpus = round(fv, 2),
        advice_text      = advice,
    )


def generate_from_params(
    age: int           = 30,
    income: float      = 80_000,
    expenses: float    = 50_000,
    savings: float     = 200_000,
    horizon: int       = 10,
    risk: str          = "moderate",
) -> dict:
    """
    Convenience function that accepts raw params instead of a UserProfile.
    Returns a JSON-serialisable dict.
    """
    risk_map = {
        "conservative": [1, 1, 2],
        "moderate":     [2, 2, 2],
        "aggressive":   [3, 3, 3],
    }
    profile = UserProfile(
        age              = age,
        monthly_income   = income,
        monthly_expenses = expenses,
        current_savings  = savings,
        investment_horizon = horizon,
        risk_answers     = risk_map.get(risk, [2, 2, 2]),
    )
    rec = generate_recommendation(profile)
    return {
        "risk_tier":        rec.risk_tier,
        "portfolio_mix":    rec.portfolio_mix,
        "monthly_invest":   rec.monthly_invest,
        "products":         rec.products,
        "budgeting_tips":   rec.budgeting_tips,
        "savings_target":   rec.savings_target,
        "projected_corpus": rec.projected_corpus,
        "advice_text":      rec.advice_text,
        "savings_rate_pct": round(profile.savings_rate * 100, 1),
    }


def assess_risk_from_quiz(answers: list[int]) -> RiskTier:
    """
    Given a list of 3 quiz answers (each 1/2/3), return the risk tier.
    """
    score = sum(answers)
    if score <= 5:
        return "conservative"
    elif score <= 8:
        return "moderate"
    else:
        return "aggressive"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_advice_text(profile: UserProfile, tier: RiskTier,
                        monthly_invest: float, projected_corpus: float) -> str:
    surplus_pct = round(profile.savings_rate * 100, 1)
    return (
        f"Based on your profile (age {profile.age}, "
        f"₹{profile.monthly_income:,.0f}/month income, "
        f"{profile.investment_horizon}-year horizon), you are classified as a "
        f"**{tier.title()} investor**.\n\n"
        f"Your current savings rate is {surplus_pct}% of income. "
        f"We recommend investing ₹{monthly_invest:,.0f}/month "
        f"across the suggested portfolio mix.\n\n"
        f"At an expected annual return of "
        f"{_EXPECTED_RETURNS[tier]*100:.1f}%, your portfolio could grow to "
        f"approximately **₹{projected_corpus:,.0f}** "
        f"in {profile.investment_horizon} years.\n\n"
        f"Top tip: Start SIPs immediately — even small, consistent contributions "
        f"compound significantly over time."
    )


# ── CLI usage ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    rec = generate_from_params(
        age=28, income=90_000, expenses=55_000,
        savings=1_50_000, horizon=15, risk="moderate",
    )
    print(json.dumps(rec, indent=2, ensure_ascii=False))
