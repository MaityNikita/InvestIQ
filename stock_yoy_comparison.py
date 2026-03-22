"""
=============================================================================
 STOCK MARKET - YEAR OVER YEAR COMPARISON (Line Graph)
 IT Companies: TCS, Infosys, Wipro, HCL, Tech Mahindra + NIFTY IT Index
 Dataset: Downloaded from Kaggle using kaggle API
=============================================================================

 SETUP INSTRUCTIONS:
 1. Install dependencies:
       pip install kaggle pandas matplotlib seaborn yfinance

 2. Set up Kaggle API credentials:
    - Go to https://www.kaggle.com/account
    - Click "Create New API Token" -> downloads kaggle.json
    - Place it at: ~/.kaggle/kaggle.json  (Linux/Mac)
                   C:/Users/<YourName>/.kaggle/kaggle.json  (Windows)
    - Run: chmod 600 ~/.kaggle/kaggle.json  (Linux/Mac only)

 3. Run this script in Spyder IDE.
    If Kaggle download fails, the script AUTO-FALLS BACK to yfinance live data.
=============================================================================
"""

import os
import warnings
import zipfile
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

COMPANIES = {
    "TCS":           "TCS.NS",
    "Infosys":       "INFY.NS",
    "Wipro":         "WIPRO.NS",
    "HCL Tech":      "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS",
    "NIFTY IT":      "^CNXIT",          # NIFTY IT Index
}

START_YEAR = 2018
END_YEAR   = 2024

# Color palette — one distinct color per company
PALETTE = {
    "TCS":           "#00C9FF",
    "Infosys":       "#FF6B6B",
    "Wipro":         "#FFD93D",
    "HCL Tech":      "#6BCB77",
    "Tech Mahindra": "#C77DFF",
    "NIFTY IT":      "#FF9F1C",
}

OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "StockCharts")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Try Kaggle download, fall back to yfinance
# ─────────────────────────────────────────────────────────────────────────────

def try_kaggle_download():
    """
    Attempts to download a Kaggle NSE/BSE IT stock dataset.
    Returns a dict {company_name: DataFrame} or None on failure.
    """
    try:
        import kaggle  # noqa: F401
        dataset_slug = "rohanrao/nifty50-stock-market-data"   # popular NSE dataset
        download_path = os.path.join(OUTPUT_DIR, "kaggle_data")
        os.makedirs(download_path, exist_ok=True)

        print("📦  Downloading Kaggle dataset:", dataset_slug)
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug,
             "-p", download_path, "--unzip"],
            check=True, capture_output=True
        )
        print("✅  Kaggle download complete.")

        # The NIFTY-50 dataset has one CSV per symbol
        ticker_to_name = {v.replace(".NS", ""): k for k, v in COMPANIES.items()
                          if v != "^CNXIT"}
        data = {}
        for fname in os.listdir(download_path):
            stem = fname.replace(".csv", "").upper()
            if stem in ticker_to_name:
                df = pd.read_csv(os.path.join(download_path, fname),
                                 parse_dates=["Date"], index_col="Date")
                df.sort_index(inplace=True)
                name = ticker_to_name[stem]
                data[name] = df[["Close"]].rename(columns={"Close": name})
        if data:
            return data
    except Exception as e:
        print(f"⚠️  Kaggle download skipped ({e}). Switching to yfinance …")
    return None


def fetch_yfinance():
    """Fetches historical closing prices using yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run:  pip install yfinance")

    data = {}
    start = f"{START_YEAR}-01-01"
    end   = f"{END_YEAR}-12-31"

    for name, ticker in COMPANIES.items():
        print(f"   📡  Fetching {name} ({ticker}) …")
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
            if df.empty:
                print(f"      ⚠️  No data for {ticker}, skipping.")
                continue
            data[name] = df[["Close"]].rename(columns={"Close": name})
        except Exception as e:
            print(f"      ⚠️  {ticker} failed: {e}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Load & merge data
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  STOCK MARKET YoY COMPARISON — IT Sector")
print("="*60)

raw = try_kaggle_download()
if not raw:
    print("\n🌐  Fetching live data via yfinance …")
    raw = fetch_yfinance()

if not raw:
    raise RuntimeError("No data fetched. Check internet / credentials.")

# Merge all companies into one DataFrame
merged = pd.concat(raw.values(), axis=1)
merged = merged.loc[f"{START_YEAR}-01-01":f"{END_YEAR}-12-31"]
merged.dropna(how="all", inplace=True)
print(f"\n✅  Data loaded: {merged.shape[0]} trading days,"
      f" {merged.shape[1]} series")
print(f"    Period: {merged.index.min().date()} → {merged.index.max().date()}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Normalise to 100 at the start of each year (YoY index)
# ─────────────────────────────────────────────────────────────────────────────

years = list(range(START_YEAR, END_YEAR + 1))

# year_data[year] = DataFrame normalised to 100 on Jan-1 of that year
year_data = {}
for yr in years:
    yr_df = merged[merged.index.year == yr].copy()
    if yr_df.empty:
        continue
    base = yr_df.iloc[0]
    normed = (yr_df / base) * 100
    year_data[yr] = normed


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — PLOT 1: Full timeline (absolute close prices, log scale)
# ─────────────────────────────────────────────────────────────────────────────

def plot_full_timeline(df, companies, palette):
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    for col in companies:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        ax.plot(s.index, s.values,
                color=palette.get(col, "white"),
                linewidth=1.6, alpha=0.9, label=col)

        # Annotate last value
        ax.annotate(f"  {col}",
                    xy=(s.index[-1], s.values[-1]),
                    color=palette.get(col, "white"),
                    fontsize=8, va="center",
                    fontfamily="monospace")

    # Shade alternate years
    for yr in years:
        if yr % 2 == 0:
            ax.axvspan(pd.Timestamp(f"{yr}-01-01"),
                       pd.Timestamp(f"{yr}-12-31"),
                       color="white", alpha=0.03)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"₹{x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.tick_params(colors="#aaa", labelsize=9)
    ax.grid(axis="y", color="#222", linewidth=0.6, linestyle="--")
    ax.grid(axis="x", color="#222", linewidth=0.4, linestyle=":")

    ax.set_title("IT Sector — Historical Close Prices (Log Scale)",
                 color="white", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Year", color="#888", fontsize=10)
    ax.set_ylabel("Close Price (₹, Log)", color="#888", fontsize=10)

    legend = ax.legend(frameon=False, labelcolor="white",
                       fontsize=9, loc="upper left")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_full_timeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"💾  Saved → {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — PLOT 2: Year-over-Year overlay (normalised to 100)
# ─────────────────────────────────────────────────────────────────────────────

def plot_yoy_overlay(year_data, companies, palette):
    """
    One subplot per company.
    X-axis = day-of-year (1–365), Y-axis = indexed price (base 100 = Jan 1).
    Each line = one year, colored from cool → warm as years advance.
    """
    valid_companies = [c for c in companies if
                       any(c in yd.columns for yd in year_data.values())]
    n = len(valid_companies)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4 + 1),
                             sharex=False)
    fig.patch.set_facecolor("#0D1117")
    axes = axes.flatten()

    # Year colour ramp: blue → red
    year_colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(year_data)))

    for idx, company in enumerate(valid_companies):
        ax = axes[idx]
        ax.set_facecolor("#0D1117")

        for yr_idx, (yr, yd) in enumerate(sorted(year_data.items())):
            if company not in yd.columns:
                continue
            s = yd[company].dropna()
            # Convert date index → day-of-year
            doy = s.index.dayofyear
            ax.plot(doy, s.values,
                    color=year_colors[yr_idx],
                    linewidth=1.4, alpha=0.85,
                    label=str(yr))

        ax.axhline(100, color="#555", linewidth=0.8, linestyle="--")

        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.tick_params(colors="#aaa", labelsize=8)
        ax.grid(color="#1e1e1e", linewidth=0.5)
        ax.set_title(company, color=palette.get(company, "white"),
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Day of Year", color="#666", fontsize=8)
        ax.set_ylabel("Indexed (Jan 1 = 100)", color="#666", fontsize=8)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    # Shared year legend
    legend_elements = [
        Line2D([0], [0], color=year_colors[i], lw=2, label=str(yr))
        for i, yr in enumerate(sorted(year_data.keys()))
    ]
    fig.legend(handles=legend_elements, title="Year",
               title_fontsize=10, fontsize=9,
               frameon=False, labelcolor="white",
               loc="lower right", ncol=len(year_data),
               bbox_to_anchor=(0.98, 0.01))

    fig.suptitle("Year-over-Year Performance — IT Companies (Indexed, Jan 1 = 100)",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_yoy_overlay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"💾  Saved → {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — PLOT 3: Annual Return Bar + Line Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_annual_returns(merged, companies, palette):
    """Bar chart of annual % returns with a trend line overlay."""
    annual_returns = {}
    for yr in years:
        yr_df = merged[merged.index.year == yr]
        if len(yr_df) < 2:
            continue
        ret = {}
        for col in companies:
            if col not in yr_df.columns:
                continue
            first = yr_df[col].dropna().iloc[0] if not yr_df[col].dropna().empty else np.nan
            last  = yr_df[col].dropna().iloc[-1] if not yr_df[col].dropna().empty else np.nan
            ret[col] = ((last - first) / first * 100) if (first and not np.isnan(first)) else np.nan
        annual_returns[yr] = ret

    ret_df = pd.DataFrame(annual_returns).T   # rows=years, cols=companies

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    x     = np.arange(len(ret_df.index))
    width = 0.12
    offsets = np.linspace(-(len(companies)-1)/2,
                           (len(companies)-1)/2,
                           len(companies)) * width

    for i, col in enumerate(companies):
        if col not in ret_df.columns:
            continue
        vals = ret_df[col].values
        bars = ax.bar(x + offsets[i], vals, width=width,
                      color=palette.get(col, "grey"), alpha=0.85,
                      label=col, zorder=3)
        # Trend line
        valid = ~np.isnan(vals)
        if valid.sum() > 1:
            ax.plot(x[valid] + offsets[i], vals[valid],
                    color=palette.get(col, "grey"),
                    linewidth=1.2, linestyle="--", alpha=0.6, zorder=4)

    ax.axhline(0, color="#555", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(ret_df.index, color="#aaa", fontsize=10)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.tick_params(colors="#aaa")
    ax.grid(axis="y", color="#1e1e1e", linewidth=0.6, linestyle="--", zorder=0)

    ax.set_title("Annual Returns (%) — IT Companies Year-over-Year",
                 color="white", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Year", color="#888", fontsize=10)
    ax.set_ylabel("Annual Return (%)", color="#888", fontsize=10)

    legend = ax.legend(frameon=False, labelcolor="white",
                       fontsize=9, loc="upper left", ncol=2)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_annual_returns.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"💾  Saved → {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 — PLOT 4: Cumulative Growth (since START_YEAR)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cumulative_growth(merged, companies, palette):
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    for col in companies:
        if col not in merged.columns:
            continue
        s = merged[col].dropna()
        cum = (s / s.iloc[0]) * 100
        ax.plot(cum.index, cum.values,
                color=palette.get(col, "white"),
                linewidth=2.0, alpha=0.9, label=col)
        # Final value annotation
        ax.annotate(f" {col}\n {cum.values[-1]:.0f}",
                    xy=(cum.index[-1], cum.values[-1]),
                    color=palette.get(col, "white"),
                    fontsize=7.5, va="center",
                    fontfamily="monospace")

    ax.axhline(100, color="#555", linewidth=0.9, linestyle="--",
               label=f"Base = {START_YEAR} open")

    for yr in years:
        ax.axvline(pd.Timestamp(f"{yr}-01-01"),
                   color="#2a2a2a", linewidth=0.8, linestyle=":")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{y:.0f}"))

    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.tick_params(colors="#aaa", labelsize=9)
    ax.grid(color="#1a1a1a", linewidth=0.6, linestyle="--")

    ax.set_title(f"Cumulative Growth — IT Companies ({START_YEAR}–{END_YEAR})\n"
                 f"Base = 100 on {START_YEAR} Jan 1",
                 color="white", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Year", color="#888", fontsize=10)
    ax.set_ylabel("Indexed Growth (Base = 100)", color="#888", fontsize=10)

    legend = ax.legend(frameon=False, labelcolor="white",
                       fontsize=9, loc="upper left")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_cumulative_growth.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"💾  Saved → {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  RUN ALL PLOTS
# ─────────────────────────────────────────────────────────────────────────────

all_companies = list(COMPANIES.keys())

print("\n📊  Generating plots …\n")

plot_full_timeline(merged, all_companies, PALETTE)
plot_yoy_overlay(year_data, all_companies, PALETTE)
plot_annual_returns(merged, all_companies, PALETTE)
plot_cumulative_growth(merged, all_companies, PALETTE)

print("\n" + "="*60)
print(f"  ✅  All 4 charts saved to: {OUTPUT_DIR}")
print("="*60 + "\n")
