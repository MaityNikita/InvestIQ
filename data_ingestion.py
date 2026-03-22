"""
=============================================================================
  data_ingestion.py — Data Download & Preprocessing Module
  AI Financial Advisory Chatbot — InvestIQ
=============================================================================
  Handles all data acquisition:
    - yfinance live data download
    - Kaggle dataset fallback
    - Basic cleaning and validation
=============================================================================
"""

import os
import warnings
import subprocess
import pandas as pd

warnings.filterwarnings("ignore")

# ── Default IT-sector companies (NSE India) ───────────────────────────────────
DEFAULT_COMPANIES = {
    "TCS":           "TCS.NS",
    "Infosys":       "INFY.NS",
    "Wipro":         "WIPRO.NS",
    "HCL Tech":      "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS",
    "NIFTY IT":      "^CNXIT",
}

DEFAULT_START = "2018-01-01"
DEFAULT_END   = "2024-12-31"


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def download_data(
    companies: dict = None,
    start: str = DEFAULT_START,
    end: str   = DEFAULT_END,
    output_dir: str = None,
    verbose: bool = True,
) -> dict:
    """
    Download OHLCV data for a dict of {name: ticker} pairs.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping company name → cleaned OHLCV DataFrame (indexed by date).
    """
    if companies is None:
        companies = DEFAULT_COMPANIES

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Try Kaggle first, fall back to yfinance
    data = _try_kaggle(companies, output_dir, verbose)
    if not data:
        data = _fetch_yfinance(companies, start, end, verbose)

    if not data:
        raise RuntimeError(
            "No data fetched. Check your internet connection or Kaggle credentials."
        )

    # Post-processing: ensure OHLCV columns, drop nulls
    cleaned = {}
    for name, df in data.items():
        df = _clean(df)
        if df.empty:
            if verbose:
                print(f"  ⚠️  {name} — empty after cleaning, skipping.")
            continue
        cleaned[name] = df
        if verbose:
            print(f"  ✅  {name:15s} — {len(df)} trading days "
                  f"({df.index.min().date()} → {df.index.max().date()})")

    return cleaned


def get_single_ticker(ticker: str, start: str = DEFAULT_START,
                       end: str = DEFAULT_END) -> pd.DataFrame:
    """
    Convenience function: download a single ticker and return a clean DataFrame.
    """
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    return _clean(df)


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_yfinance(companies: dict, start: str, end: str,
                    verbose: bool) -> dict:
    """Download data for all companies via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run:  pip install yfinance")

    data = {}
    if verbose:
        print("\n📡  Downloading via yfinance …")

    for name, ticker in companies.items():
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
            if not df.empty:
                data[name] = df
            elif verbose:
                print(f"  ⚠️  {name} ({ticker}) — no data returned.")
        except Exception as exc:
            if verbose:
                print(f"  ⚠️  {name} ({ticker}) failed: {exc}")

    return data


def _try_kaggle(companies: dict, output_dir: str | None,
                verbose: bool) -> dict | None:
    """
    Attempt to pull an NSE dataset from Kaggle.
    Returns None on any failure so the caller can fall back to yfinance.
    """
    try:
        import kaggle  # noqa: F401
        dataset_slug  = "rohanrao/nifty50-stock-market-data"
        download_path = output_dir or os.path.join(
            os.path.expanduser("~"), "kaggle_cache"
        )
        os.makedirs(download_path, exist_ok=True)

        if verbose:
            print(f"📦  Attempting Kaggle download: {dataset_slug}")

        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug,
             "-p", download_path, "--unzip"],
            check=True, capture_output=True
        )

        # Map ticker stem → company name
        ticker_map = {v.replace(".NS", "").upper(): k
                      for k, v in companies.items() if v != "^CNXIT"}
        data = {}
        for fname in os.listdir(download_path):
            stem = fname.replace(".csv", "").upper()
            if stem in ticker_map:
                df = pd.read_csv(
                    os.path.join(download_path, fname),
                    parse_dates=["Date"], index_col="Date"
                )
                df.sort_index(inplace=True)
                data[ticker_map[stem]] = df

        if data and verbose:
            print("  ✅  Kaggle download complete.")
        return data or None

    except Exception as exc:
        if verbose:
            print(f"  ℹ️  Kaggle skipped ({type(exc).__name__}). "
                  f"Falling back to yfinance …")
        return None


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has standard OHLCV columns, a DatetimeIndex,
    no NaNs, and no duplicate dates.
    """
    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only OHLCV
    needed = ["Open", "High", "Low", "Close", "Volume"]
    present = [c for c in needed if c in df.columns]
    df = df[present].copy()

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    df.dropna(subset=["Close"], inplace=True)
    df.fillna(method="ffill", inplace=True)

    return df


# ── CLI usage ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = download_data(verbose=True)
    for name, df in data.items():
        print(f"  {name}: {df.shape}")
