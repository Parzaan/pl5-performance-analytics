"""Data loading, caching, and normalization helpers for the PL5 player dashboard."""

from pathlib import Path

import pandas as pd
import streamlit as st

# Repo root is one level up from this file (dashboard/utils.py -> repo root)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "merged.csv"

RADAR_FEATURES = {
    "shots_p90": "Shots/90",
    "shot_accuracy": "Shot acc.",
    "assists_p90": "Assists/90",
    "goals_excl_pk_p90": "Goals/90",
    "minutes": "Minutes",
}


@st.cache_data
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the processed player-season table produced by 01_eda.ipynb."""
    df = pd.read_csv(path)

    # primary_pos is added in the EDA notebook, but recompute defensively
    # in case an older version of merged.csv is loaded.
    if "primary_pos" not in df.columns:
        df["primary_pos"] = df["pos"].astype(str).str.split(",").str[0]

    return df


@st.cache_data
def radar_axis_maxima(df: pd.DataFrame) -> dict:
    """Per-feature maxima used to normalize the radar chart onto a 0-1 scale."""
    return {col: float(df[col].max()) for col in RADAR_FEATURES}


@st.cache_data
def league_average_vector(df: pd.DataFrame, maxima: dict) -> list:
    """League-wide average profile, normalized the same way as a player's vector."""
    return [
        float(df[col].mean()) / maxima[col] if maxima[col] else 0.0
        for col in RADAR_FEATURES
    ]


def player_vector(row: pd.Series, maxima: dict) -> list:
    """Normalize one player's row onto the same 0-1 radar scale.

    Ratio stats (e.g. shot_accuracy = SoT/Sh) are undefined (NaN) for players
    with 0 shots -- treat that as 0 on the radar rather than letting NaN break
    the whole shape.
    """
    vec = []
    for col in RADAR_FEATURES:
        val = row[col]
        if pd.isna(val) or not maxima[col]:
            vec.append(0.0)
        else:
            vec.append(float(val) / maxima[col])
    return vec


def has_shot_quality_data(row: pd.Series) -> bool:
    """False for players with 0 shots this season (G/Sh is undefined, not 0)."""
    return pd.notna(row.get("goals_per_shot")) and pd.notna(row.get("shots_p90"))


def filter_players(
    df: pd.DataFrame,
    leagues: list[str] | None = None,
    positions: list[str] | None = None,
    season: str | None = None,
    min_minutes: int = 450,
) -> pd.DataFrame:
    """Apply the sidebar filters shared by the radar and scatter views."""
    out = df[df["minutes"] >= min_minutes]
    if leagues:
        out = out[out["league"].isin(leagues)]
    if positions:
        out = out[out["primary_pos"].isin(positions)]
    if season:
        out = out[out["season"] == season]
    return out

ELIGIBLE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "phase_c_eligible_players.csv"
FORECAST_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "forecast_results.csv"


@st.cache_data
def load_eligible_players(path: Path = ELIGIBLE_PATH) -> list[str]:
    """The 264 Phase C eligible players — the only players the forecast view covers."""
    return pd.read_csv(path)["player_key"].tolist()


@st.cache_data
def load_forecast_results(path: Path = FORECAST_PATH) -> pd.DataFrame:
    """Predicted vs. actual goals_excl_pk_p90 for the 2024-25 season, from 03_forecasting.ipynb."""
    return pd.read_csv(path)

HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "player_seasons_merged.csv"
PRIOR_SEASONS = ["1920", "2021", "2122", "2223", "2324"]


@st.cache_data
def load_player_history(path: Path = HISTORY_PATH) -> pd.DataFrame:
    """Player-season rows for the 5 prior seasons used as model input (2019-20 to 2023-24)."""
    df = pd.read_csv(path, dtype={"season": str})
    return df[df["season"].isin(PRIOR_SEASONS)]

def format_season_label(season_code: str) -> str:
    """Convert a compact season code like '1920' into a readable '2019-20' label."""
    start_year = "20" + season_code[:2]
    end_year_suffix = season_code[2:]
    return f"{start_year}-{end_year_suffix}"