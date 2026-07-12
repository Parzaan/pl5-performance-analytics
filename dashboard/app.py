"""PL5 Player Analytics — scouting dashboard.

Run from the repo root with:
    streamlit run dashboard/app.py
"""

import streamlit as st

from components import build_radar, build_scatter
from utils import (
    filter_players,
    has_shot_quality_data,
    league_average_vector,
    load_data,
    radar_axis_maxima,
)

st.set_page_config(page_title="PL5 Player Analytics", layout="wide")

st.title("Player scouting dashboard")
st.caption(
    "Big 5 European leagues · 2019-20 to 2024-25 · FBref public data · "
    "non-penalty goals/90 as the core attacking-output measure"
)

df = load_data()

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Filters")

leagues = st.sidebar.multiselect(
    "League",
    options=sorted(df["league"].unique()),
    default=[],
    help="Leave empty to include every league.",
)
positions = st.sidebar.multiselect(
    "Position",
    options=sorted(df["primary_pos"].dropna().unique()),
    default=[],
    help="Leave empty to include every position.",
)

season_options = sorted(df["season"].unique(), reverse=True)
season = st.sidebar.selectbox(
    "Season",
    options=season_options,
    index=0,
    help="A player-season table has one row per player per season — "
    "picking a single season keeps the radar/scatter to one point per player.",
)

min_minutes = st.sidebar.slider("Minimum minutes played", 0, 3000, 450, step=50)

filtered = filter_players(
    df, leagues=leagues, positions=positions, season=season, min_minutes=min_minutes
)

if filtered.empty:
    st.warning("No players match the current filters — widen the filters in the sidebar.")
    st.stop()

# ---------------------------------------------------------------- player select
player_names = sorted(filtered["player"].unique())
selected_player = st.selectbox("Player", options=player_names)

player_row = filtered[filtered["player"] == selected_player].iloc[0]

# ---------------------------------------------------------------- layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Player profile")
    maxima = radar_axis_maxima(filtered)
    league_avg = league_average_vector(filtered, maxima)
    st.plotly_chart(build_radar(player_row, maxima, league_avg), use_container_width=True)
    st.caption("Red = selected player · blue = average across the current filter selection.")

with col2:
    st.subheader("Shots vs. finishing quality")
    if not has_shot_quality_data(player_row):
        st.info(
            f"{selected_player} recorded 0 shots this season, so the shot-quality "
            "rate (G/Sh) is undefined — not shown on the chart below."
        )
    st.plotly_chart(build_scatter(filtered, selected_player), use_container_width=True)
    st.caption("Each dot is a player-season passing the filters; the star is the selected player.")

# ---------------------------------------------------------------- stat line
st.divider()
metric_cols = st.columns(5)
metric_cols[0].metric("Goals/90 (non-pen)", f"{player_row['goals_excl_pk_p90']:.2f}")
metric_cols[1].metric("Shots/90", f"{player_row['shots_p90']:.2f}")
metric_cols[2].metric("Shot accuracy", f"{player_row['shot_accuracy']:.0f}%")
metric_cols[3].metric("Assists/90", f"{player_row['assists_p90']:.2f}")
metric_cols[4].metric("Minutes", f"{int(player_row['minutes']):,}")