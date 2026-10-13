"""
Generate one to five months of synthetic FMCG sales data for UI / MMM testing.

Each month is saved as a separate Excel file with a distinct column schema
(date format, available channels, column naming). Sales are driven by a
simple latent media-mix structure (base demand + channel effects + noise)
without fitting any model.

Run directly:
    python test_streamlit.py

Or with Streamlit:
    streamlit run test_streamlit.py
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent / "synthetic_data"
RANDOM_SEED = 42

# Latent MMM-style coefficients (units sold per $1k spend, roughly scaled)
CHANNEL_COEFFICIENTS = {
    "TV_spend": 0.85,
    "Meta_spend": 1.10,
    "Youtube_spend": 0.95,
    "PR_spend": 0.45,
    "Instagram_spend": 1.05,
}

BASE_DAILY_SALES = 12_500.0
SALES_NOISE_STD = 450.0

# Five consecutive calendar months used across all files
START_YEAR = 2025
START_MONTH = 5  # May 2025 through September 2025


# ---------------------------------------------------------------------------
# Date formatting helpers
# ---------------------------------------------------------------------------
def short_date(d: date) -> str:
    """Short date form, e.g. 5/1/25 (no leading zeros)."""
    return f"{d.month}/{d.day}/{str(d.year)[-2:]}"


def long_date(d: date) -> str:
    """Long date form, e.g. May 1, 2025."""
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def month_date_range(year: int, month: int) -> list[date]:
    """Return every calendar day in the given month."""
    _, days_in_month = calendar.monthrange(year, month)
    return [date(year, month, day) for day in range(1, days_in_month + 1)]


# ---------------------------------------------------------------------------
# Synthetic spend and sales generation
# ---------------------------------------------------------------------------
def _campaign_pulse(day_index: int, period: int, amplitude: float) -> float:
    """Smooth periodic pulse mimicking flighted media."""
    phase = 2 * np.pi * day_index / period
    return amplitude * (0.5 + 0.5 * np.sin(phase))


def generate_daily_spends(
    dates: list[date],
    rng: np.random.Generator,
    *,
    include_instagram: bool,
) -> dict[str, np.ndarray]:
    """
    Build daily media spends with realistic FMCG patterns:
    - TV flighted in bursts with higher weekend weighting
    - Meta / Instagram relatively always-on with correlated bursts
    - YouTube tied to content-release weeks
    - PR low baseline with occasional event spikes
    """
    n = len(dates)
    day_idx = np.arange(n)

    # Weekend index: Fri–Sun get a small lift in retail media activity
    weekend = np.array([d.weekday() >= 4 for d in dates], dtype=float)

    tv = (
        18_000
        + 6_000 * _campaign_pulse(day_idx, period=10, amplitude=1.0)
        + 2_500 * weekend
        + rng.normal(0, 800, n)
    )
    meta = (
        9_500
        + 2_000 * _campaign_pulse(day_idx, period=7, amplitude=1.0)
        + rng.normal(0, 400, n)
    )
    youtube = (
        5_500
        + 3_500 * _campaign_pulse(day_idx, period=14, amplitude=1.0)
        + rng.normal(0, 350, n)
    )
    pr = (
        1_200
        + 800 * (rng.random(n) < 0.12).astype(float)  # ~12% PR event days
        + rng.normal(0, 120, n)
    )

    spends = {
        "TV_spend": np.clip(tv, 500, None),
        "Meta_spend": np.clip(meta, 200, None),
        "Youtube_spend": np.clip(youtube, 100, None),
        "PR_spend": np.clip(pr, 0, None),
    }

    if include_instagram:
        # Instagram partially follows Meta but with its own creative flights
        instagram = (
            0.55 * spends["Meta_spend"]
            + 2_800
            + 1_200 * _campaign_pulse(day_idx + 2, period=9, amplitude=1.0)
            + rng.normal(0, 300, n)
        )
        spends["Instagram_spend"] = np.clip(instagram, 150, None)

    return spends


def generate_sales(
    dates: list[date],
    spends: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sales = intercept + sum(channel contribution) + weekly seasonality + noise.

    Uses sqrt(spend) for mild diminishing returns — common in MMM data prep
    without fitting a model here.
    """
    n = len(dates)
    sales = np.full(n, BASE_DAILY_SALES, dtype=float)

    for channel, coef in CHANNEL_COEFFICIENTS.items():
        if channel not in spends:
            continue
        # Scale spend from dollars to thousands before sqrt transform
        sales += coef * np.sqrt(spends[channel] / 1_000.0)

    # FMCG weekly seasonality: higher basket size Thu–Sun
    dow_weights = (0, 50, 80, 120, 180, 220, 160)
    dow_lift = np.array([dow_weights[d.weekday()] for d in dates], dtype=float)
    sales += dow_lift

    # Slow macro drift across the month
    sales += np.linspace(0, 300, n)

    sales += rng.normal(0, SALES_NOISE_STD, n)
    return np.clip(sales, 1_000, None)


def build_month_dataframe(
    year: int,
    month: int,
    rng: np.random.Generator,
    *,
    date_formatter,
    columns: list[str],
    pr_column_name: str = "PR_spend",
    include_instagram: bool = False,
    duplicate_youtube: bool = False,
) -> pd.DataFrame:
    """Assemble one month of synthetic data with the requested schema."""
    dates = month_date_range(year, month)
    spends = generate_daily_spends(dates, rng, include_instagram=include_instagram)
    sales = generate_sales(dates, spends, rng)

    value_lookup: dict[str, list] = {
        "date": [date_formatter(d) for d in dates],
        "sales": np.round(sales, 2).tolist(),
        "TV_spend": np.round(spends["TV_spend"], 2).tolist(),
        "Meta_spend": np.round(spends["Meta_spend"], 2).tolist(),
        "Youtube_spend": np.round(spends["Youtube_spend"], 2).tolist(),
        pr_column_name: np.round(spends["PR_spend"], 2).tolist(),
        "PR_spend": np.round(spends["PR_spend"], 2).tolist(),
    }

    if include_instagram:
        value_lookup["Instagram_spend"] = np.round(spends["Instagram_spend"], 2).tolist()

    # Build column-by-column so duplicate names (e.g. two Youtube_spend) are kept
    series_list: list[pd.Series] = []
    youtube_seen = 0
    for col in columns:
        if col == "Youtube_spend":
            youtube_seen += 1
            if duplicate_youtube and youtube_seen == 2:
                varied = np.round(
                    spends["Youtube_spend"] * rng.uniform(0.97, 1.03, len(dates)), 2
                )
                series_list.append(pd.Series(varied, name=col))
            else:
                series_list.append(pd.Series(value_lookup["Youtube_spend"], name=col))
        else:
            series_list.append(pd.Series(value_lookup[col], name=col))

    return pd.concat(series_list, axis=1)


def generate_all_files(
    output_dir: Path | None = None,
    file_count: int = 5,
) -> list[Path]:
    """
    Create Excel files — one per month — with distinct column layouts.

    file_count must be between 1 and 5 (inclusive).

    Returns the list of written file paths.
    """
    if not 1 <= file_count <= 5:
        raise ValueError("file_count must be between 1 and 5.")
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)
    written: list[Path] = []

    month_specs = [
        {
            "filename": "fmcg_month_01.xlsx",
            "month_offset": 0,
            "date_formatter": short_date,
            "columns": [
                "date",
                "sales",
                "TV_spend",
                "Meta_spend",
                "Youtube_spend",
                "PR_spend",
                "Youtube_spend",
                "Instagram_spend",
            ],
            "duplicate_youtube": True,
            "include_instagram": True,
        },
        {
            "filename": "fmcg_month_02.xlsx",
            "month_offset": 1,
            "date_formatter": short_date,
            "columns": [
                "date",
                "sales",
                "TV_spend",
                "Meta_spend",
                "Youtube_spend",
                "PR_spend",
                "Instagram_spend",
            ],
            "include_instagram": True,
        },
        {
            "filename": "fmcg_month_03.xlsx",
            "month_offset": 2,
            "date_formatter": long_date,
            "columns": [
                "date",
                "sales",
                "TV_spend",
                "Meta_spend",
                "Youtube_spend",
                "PR_spend",
            ],
        },
        {
            "filename": "fmcg_month_04.xlsx",
            "month_offset": 3,
            "date_formatter": short_date,
            "pr_column_name": "Public Relations Spend",
            "columns": [
                "date",
                "sales",
                "TV_spend",
                "Meta_spend",
                "Youtube_spend",
                "Public Relations Spend",
            ],
        },
        {
            "filename": "fmcg_month_05.xlsx",
            "month_offset": 4,
            "date_formatter": short_date,
            "columns": [
                "date",
                "sales",
                "TV_spend",
                "Meta_spend",
                "Youtube_spend",
                "PR_spend",
            ],
        },
    ]

    for spec in month_specs[:file_count]:
        month = START_MONTH + spec["month_offset"]
        year = START_YEAR + (month - 1) // 12
        month = ((month - 1) % 12) + 1

        df = build_month_dataframe(
            year,
            month,
            rng,
            date_formatter=spec["date_formatter"],
            columns=spec["columns"],
            pr_column_name=spec.get("pr_column_name", "PR_spend"),
            include_instagram=spec.get("include_instagram", False),
            duplicate_youtube=spec.get("duplicate_youtube", False),
        )

        path = out / spec["filename"]
        df.to_excel(path, index=False, engine="openpyxl")
        written.append(path)

    return written


# ---------------------------------------------------------------------------
# Streamlit UI (optional)
# ---------------------------------------------------------------------------
def run_streamlit_app() -> None:
    st.set_page_config(page_title="FMCG synthetic data generator", layout="centered")
    st.title("FMCG synthetic sales data generator")
    st.markdown(
        "Creates **1 to 5 Excel files** with varied column schemas for testing the "
        "Marketing Mix Model upload UI. No models are fit — only synthetic data."
    )

    st.subheader("Output settings")
    file_count = st.number_input(
        "Number of files to generate",
        min_value=1,
        max_value=5,
        value=5,
        step=1,
        help="Generate between 1 and 5 monthly Excel files.",
    )
    custom_dir = st.text_input(
        "Output folder",
        value=str(OUTPUT_DIR),
        help="Directory where the .xlsx files will be written.",
    )

    if st.button("Generate test files", type="primary"):
        try:
            paths = generate_all_files(Path(custom_dir), file_count=int(file_count))
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            return

        st.success(f"Created {len(paths)} files.")
        for path in paths:
            st.write(f"- `{path.name}` → `{path}`")

        with st.expander("Preview month 1 (first 5 rows)"):
            preview = pd.read_excel(paths[0], engine="openpyxl")
            st.dataframe(preview.head(), use_container_width=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _running_in_streamlit() -> bool:
    """True when the script is executed via `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _running_in_streamlit():
        run_streamlit_app()
    else:
        paths = generate_all_files()
        print(f"Generated {len(paths)} files in {OUTPUT_DIR}:")
        for path in paths:
            print(f"  {path}")
