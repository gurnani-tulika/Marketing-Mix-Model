"""
Marketing Mix Model (MMM) engine — version 1.

Upload five Excel files, collate and validate the data, preview it,
then choose 4–5 marketing channels (plus date and sales) for training prep.
No regression or MMM model is implemented in this version.
"""

import calendar
from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st

from model_training import train_and_evaluate

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
REQUIRED_FILE_COUNT = 5          # Exactly five Excel files must be uploaded
MIN_CHANNELS = 4                 # More than 3 channels
MAX_CHANNELS = 5                 # Less than 6 channels
MISSING_MARKER = "*"             # Placeholder when a column is absent in a file
DATE_COLUMN_CANDIDATES = ("date", "Date", "DATE")  # Common date column names


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Marketing Mix Model engine", layout="wide")
st.title("Marketing Mix Model engine")
st.markdown(
    "Upload **5 Excel (.xlsx) files**, review the collated dataset, "
    "then select **4 or 5 channel columns** plus **date** and **sales** for training prep."
)


# ---------------------------------------------------------------------------
# Helper: read one Excel file with pandas
# ---------------------------------------------------------------------------
def read_excel_file(uploaded_file: BytesIO) -> pd.DataFrame:
    """Read a single uploaded .xlsx workbook into a DataFrame."""
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    # Strip whitespace from column names for consistency across files
    df.columns = [str(col).strip() for col in df.columns]
    return df


# ---------------------------------------------------------------------------
# Helper: check whether a value is a real calendar date
# ---------------------------------------------------------------------------
def is_valid_calendar_date(value) -> bool:
    """
    Return True if value represents a date that exists on the calendar
    (e.g. rejects 2024-02-30).
    """
    if pd.isna(value) or value == MISSING_MARKER:
        return False
    try:
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, pd.Timestamp):
            dt = value.to_pydatetime()
        else:
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                return False
            dt = parsed.to_pydatetime()

        # Confirm day/month/year round-trip (catches invalid dates like Feb 30)
        year, month, day = dt.year, dt.month, dt.day
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return False
        max_day = calendar.monthrange(year, month)[1]
        return day <= max_day
    except (ValueError, TypeError, OverflowError):
        return False


# ---------------------------------------------------------------------------
# Helper: check whether a value is numeric (asterisk counts as missing, not numeric)
# ---------------------------------------------------------------------------
def is_numeric_value(value) -> bool:
    """Return True if value can be treated as a number (excludes the missing marker)."""
    if value == MISSING_MARKER or pd.isna(value):
        return False
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Helper: find the date column name across collated columns
# ---------------------------------------------------------------------------
def find_date_column(columns: list[str]) -> str | None:
    """Pick the first column whose name matches common date labels."""
    lower_map = {col.lower(): col for col in columns}
    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


# ---------------------------------------------------------------------------
# Collate multiple DataFrames into one, filling absent columns with *
# ---------------------------------------------------------------------------
def collate_dataframes(dataframes: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge DataFrames from all uploaded files.

    - Union of all column names is used.
    - Columns missing from a file are filled with MISSING_MARKER.
    - missing_mask marks every imputed cell (True = filled with *).

    Returns (collated_df, missing_mask).
    """
    all_columns = sorted({col for df in dataframes for col in df.columns})
    aligned_frames = []
    masks = []

    for df in dataframes:
        # Track which columns were absent in this source file
        absent_cols = set(all_columns) - set(df.columns)
        row_mask = pd.DataFrame(False, index=df.index, columns=all_columns)

        # Reindex to the full column set; new columns start as NaN
        aligned = df.reindex(columns=all_columns)

        for col in absent_cols:
            aligned[col] = MISSING_MARKER
            row_mask[col] = True

        aligned_frames.append(aligned)
        masks.append(row_mask)

    collated = pd.concat(aligned_frames, ignore_index=True)
    missing_mask = pd.concat(masks, ignore_index=True)
    return collated, missing_mask


# ---------------------------------------------------------------------------
# Validate collated data: dates and numeric fields
# ---------------------------------------------------------------------------
def validate_collated_data(
    df: pd.DataFrame, date_col: str | None
) -> tuple[list[str], list[str]]:
    """
    Run validation checks on the collated table.

    Returns two lists of error messages: (date_errors, numeric_errors).
    """
    date_errors: list[str] = []
    numeric_errors: list[str] = []

    # --- Date column validation ---
    if date_col and date_col in df.columns:
        invalid_dates = df[date_col].apply(lambda v: not is_valid_calendar_date(v))
        if invalid_dates.any():
            bad_count = int(invalid_dates.sum())
            date_errors.append(
                f"Column '{date_col}': {bad_count} row(s) have invalid or missing dates."
            )
    elif date_col:
        date_errors.append(f"Date column '{date_col}' was not found in the collated data.")
    else:
        date_errors.append(
            "No date column detected. Include a column named 'date' (any casing) "
            "or select one below."
        )

    # --- Numeric validation for non-date columns ---
    numeric_cols = [c for c in df.columns if c != date_col]
    for col in numeric_cols:
        non_missing = df[col] != MISSING_MARKER
        invalid = non_missing & ~df[col].apply(is_numeric_value)
        if invalid.any():
            bad_count = int(invalid.sum())
            numeric_errors.append(
                f"Column '{col}': {bad_count} row(s) contain non-numeric values."
            )

    return date_errors, numeric_errors


# ---------------------------------------------------------------------------
# Style the preview table: red background for imputed (*) cells
# ---------------------------------------------------------------------------
def style_preview_table(df: pd.DataFrame, missing_mask: pd.DataFrame):
    """
    Apply red highlighting to every cell that was filled with MISSING_MARKER
    because the column was absent in the source file.
    """

    def highlight_cell(row_idx: int, col_name: str, value) -> str:
        if (
            row_idx < len(missing_mask)
            and col_name in missing_mask.columns
            and missing_mask.iloc[row_idx][col_name]
        ):
            return "background-color: #ff4d4d; color: white"
        if value == MISSING_MARKER:
            return "background-color: #ff4d4d; color: white"
        return ""

    def style_row(row: pd.Series) -> list[str]:
        row_idx = row.name
        return [highlight_cell(row_idx, col, row[col]) for col in df.columns]

    return df.style.apply(style_row, axis=1)


# ---------------------------------------------------------------------------
# Validate channel / sales selections for the training table
# ---------------------------------------------------------------------------
def validate_training_selection(
    df: pd.DataFrame,
    date_col: str,
    sales_col: str,
    channel_cols: list[str],
) -> list[str]:
    """
    Ensure selected columns exist, channel count is 4–5, and values are numeric.
    Returns a list of validation error messages (empty if all good).
    """
    errors: list[str] = []

    # Channel count: more than 3 and less than 6
    if len(channel_cols) < MIN_CHANNELS:
        errors.append(
            f"Select at least {MIN_CHANNELS} channels (more than 3). "
            f"Currently selected: {len(channel_cols)}."
        )
    elif len(channel_cols) > MAX_CHANNELS:
        errors.append(
            f"Select at most {MAX_CHANNELS} channels (less than 6). "
            f"Currently selected: {len(channel_cols)}."
        )

    # Required columns must exist
    for col in [date_col, sales_col] + channel_cols:
        if col not in df.columns:
            errors.append(f"Column '{col}' does not exist in the collated data.")

    if errors:
        return errors

    # Date column must contain valid calendar dates
    invalid_dates = df[date_col].apply(lambda v: not is_valid_calendar_date(v))
    if invalid_dates.any():
        errors.append(
            f"Date column '{date_col}' has {int(invalid_dates.sum())} invalid date(s)."
        )

    # Sales and channel columns must be numeric (missing * is not allowed here)
    for col in [sales_col] + channel_cols:
        non_numeric = ~df[col].apply(is_numeric_value)
        if non_numeric.any():
            errors.append(
                f"Column '{col}' has {int(non_numeric.sum())} non-numeric value(s). "
                "All training rows must be numeric."
            )

    return errors


# ---------------------------------------------------------------------------
# Build final_data: user-selected date, sales, and channel columns
# ---------------------------------------------------------------------------
def build_final_data(
    df: pd.DataFrame, date_col: str, sales_col: str, channel_cols: list[str]
) -> pd.DataFrame:
    """Collate selected columns into final_data for future MMM use."""
    cols = [date_col, sales_col] + channel_cols
    final_data = df[cols].copy()
    final_data[date_col] = pd.to_datetime(final_data[date_col], errors="coerce")
    for col in [sales_col] + channel_cols:
        final_data[col] = pd.to_numeric(final_data[col], errors="coerce")
    final_data = final_data.sort_values(by=date_col, ascending=True).reset_index(drop=True)
    return final_data


def summarize_final_data(
    final_data: pd.DataFrame,
    date_col: str,
    sales_col: str,
    channel_cols: list[str],
) -> tuple[dict[str, int], list[str], dict[str, int]]:
    """
    Summarize final_data for user validation.

    Returns (values_per_channel, channels_considered, missing_by_column).
    """
    channels_considered = list(channel_cols)
    values_per_channel = {
        col: int(final_data[col].notna().sum()) for col in channel_cols
    }

    missing_by_column: dict[str, int] = {}
    for col in [date_col, sales_col] + channel_cols:
        n_missing = int(final_data[col].isna().sum())
        if n_missing > 0:
            missing_by_column[col] = n_missing

    return values_per_channel, channels_considered, missing_by_column


# ---------------------------------------------------------------------------
# Build the training-ready subset sorted by date (display formatting)
# ---------------------------------------------------------------------------
def build_training_table(final_data: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Return final_data with the date column formatted for display."""
    training = final_data.copy()
    training[date_col] = training[date_col].dt.strftime("%Y-%m-%d")
    return training


# =============================================================================
# MAIN UI
# =============================================================================

# --- Step 1: Upload exactly five Excel files ---
st.header("1. Upload Excel files")
uploaded_files = st.file_uploader(
    "Choose exactly 5 Excel (.xlsx) files",
    type=["xlsx"],
    accept_multiple_files=True,
    help="All five files are required before collation runs.",
)

if not uploaded_files:
    st.info("Upload 5 Excel files to continue.")
    st.stop()

if len(uploaded_files) != REQUIRED_FILE_COUNT:
    st.warning(
        f"Please upload exactly {REQUIRED_FILE_COUNT} files. "
        f"You have uploaded {len(uploaded_files)}."
    )
    st.stop()

# --- Step 2: Read and collate ---
st.header("2. Collated data preview")

try:
    dataframes = [read_excel_file(f) for f in uploaded_files]
except Exception as exc:
    st.error(f"Failed to read one or more Excel files: {exc}")
    st.stop()

collated_df, missing_mask = collate_dataframes(dataframes)
detected_date_col = find_date_column(list(collated_df.columns))

# Sort by date when a date column is available (keep missing_mask in sync)
if detected_date_col:
    sort_keys = pd.to_datetime(collated_df[detected_date_col], errors="coerce")
    sort_order = sort_keys.argsort(kind="mergesort")
    collated_df = collated_df.iloc[sort_order].reset_index(drop=True)
    missing_mask = missing_mask.iloc[sort_order].reset_index(drop=True)

date_errors, numeric_errors = validate_collated_data(collated_df, detected_date_col)

# Show validation summary
if date_errors or numeric_errors:
    st.subheader("Validation notes")
    for msg in date_errors + numeric_errors:
        st.warning(msg)
else:
    st.success("All columns passed basic validation (dates and numeric values).")

st.caption(
    "Cells highlighted in **red** were filled with `*` because the column "
    "was missing in the source Excel file."
)
st.dataframe(style_preview_table(collated_df, missing_mask), width="stretch")

# --- Step 3: Select date, sales, and channels ---
st.header("3. Select columns for MMM training prep")

all_columns = list(collated_df.columns)
default_date = detected_date_col or (all_columns[0] if all_columns else "")

# Guess a sales column if one exists
sales_guess = next(
    (c for c in all_columns if c.lower() in ("sales", "revenue", "units_sold")),
    all_columns[1] if len(all_columns) > 1 else "",
)

date_col = st.selectbox(
    "Date column",
    options=all_columns,
    index=all_columns.index(default_date) if default_date in all_columns else 0,
)

sales_col = st.selectbox(
    "Sales column",
    options=all_columns,
    index=all_columns.index(sales_guess) if sales_guess in all_columns else 0,
)

# Channels: every column except date and sales
channel_options = [c for c in all_columns if c not in (date_col, sales_col)]
channel_cols = st.multiselect(
    f"Channel columns (select {MIN_CHANNELS} or {MAX_CHANNELS})",
    options=channel_options,
    help=f"Choose more than 3 and less than 6 channels ({MIN_CHANNELS}–{MAX_CHANNELS} total).",
)

# --- Step 4: Validate selection and show training table ---
st.header("4. Training data preview")

selection_errors = validate_training_selection(
    collated_df, date_col, sales_col, channel_cols
)

if selection_errors:
    for msg in selection_errors:
        st.error(msg)
else:
    final_data = build_final_data(collated_df, date_col, sales_col, channel_cols)
    values_per_channel, channels_considered, missing_by_column = summarize_final_data(
        final_data, date_col, sales_col, channel_cols
    )

    st.subheader("Final data validation")
    st.markdown("**Channels that will be considered:**")
    for channel in channels_considered:
        st.write(f"- {channel}")

    st.markdown("**Number of values under each channel:**")
    channel_counts = pd.DataFrame(
        {
            "Channel": list(values_per_channel.keys()),
            "Value count": list(values_per_channel.values()),
        }
    )
    st.dataframe(channel_counts, width="stretch", hide_index=True)

    if missing_by_column:
        st.warning("Missing values were found in the final dataset:")
        for col, count in missing_by_column.items():
            st.write(f"- `{col}`: {count} missing value(s)")
    else:
        st.success("No missing values in the final dataset.")

    training_df = build_training_table(final_data, date_col)
    st.success(
        f"Ready: {len(channel_cols)} channels, date column '{date_col}', "
        f"sales column '{sales_col}'."
    )
    st.dataframe(training_df, width="stretch")
    st.caption(
        "This table contains date, sales, and your selected channel columns "
        "sorted in ascending date order."
    )

    st.header("5. Model evaluation")

    try:
        _, evaluation = train_and_evaluate(
            final_data, date_col=date_col, sales_col=sales_col
        )

        st.subheader("Train / test split")
        st.write(
            f"Training rows: **{evaluation['train_size']}** · "
            f"Test rows: **{evaluation['test_size']}** · "
            f"Ridge alpha: **{evaluation['ridge_alpha']}**"
        )

        st.subheader("Accuracy metrics")
        metric_names = ["mse", "rmse", "mae", "r2", "mape"]
        metrics_df = pd.DataFrame(
            {
                "Metric": [name.upper() for name in metric_names],
                "Training": [
                    evaluation["train_metrics"][name] for name in metric_names
                ],
                "Test": [evaluation["test_metrics"][name] for name in metric_names],
            }
        )
        st.dataframe(
            metrics_df.style.format({"Training": "{:,.4f}", "Test": "{:,.4f}"}),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Model coefficients (scaled features)")
        coefficients_df = pd.DataFrame(
            {
                "Feature": ["intercept"]
                + list(evaluation["coefficients"].keys()),
                "Coefficient": [evaluation["intercept"]]
                + list(evaluation["coefficients"].values()),
            }
        )
        st.dataframe(
            coefficients_df.style.format({"Coefficient": "{:,.4f}"}),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Test set predictions")
        test_predictions = evaluation["test_predictions"].copy()
        test_predictions[date_col] = pd.to_datetime(
            test_predictions[date_col]
        ).dt.strftime("%Y-%m-%d")
        st.dataframe(
            test_predictions.style.format(
                {
                    "actual_sales": "{:,.2f}",
                    "predicted_sales": "{:,.2f}",
                    "residual": "{:,.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    except ValueError as exc:
        st.error(f"Model training failed: {exc}")
