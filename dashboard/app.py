"""PL5 Player Analytics — scouting dashboard.

Run from the repo root with:
    streamlit run dashboard/app.py
"""

import streamlit as st

from components import (
    build_radar,
    build_scatter,
    build_history_chart,
    build_minutes_chart,
    build_forecast_chart,
    build_gap_distribution_chart,
    build_full_forecast_scatter,
)
from utils import (
    filter_players,
    format_season_label,
    has_shot_quality_data,
    league_average_vector,
    load_data,
    load_eligible_players,
    load_forecast_results,
    load_player_history,
    radar_axis_maxima,
)

st.set_page_config(page_title="PL5 Player Analytics", layout="wide")
df = load_data()
all_player_names = sorted(df["player"].unique())
eligible_players = load_eligible_players()
eligible_names = {pk.rsplit("_", 1)[0] for pk in eligible_players}

if "selected_player_name" not in st.session_state:
    st.session_state.selected_player_name = all_player_names[0]

view = st.sidebar.radio("View", ["Scouting comparison", "Forecast results"])
st.sidebar.divider()

if view == "Scouting comparison":
    st.title("Player scouting dashboard")
    st.caption(
        "Big 5 European leagues · 2019-20 to 2024-25 · FBref public data · "
        "non-penalty goals/90 as the core attacking-output measure"
    )
else:
    st.title("Forecast results dashboard")
    st.caption(
        "Retrospective check of a 2024-25 non-penalty goals/90 forecasting model, "
        "built on five prior seasons as input"
    )

if view == "Scouting comparison":
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
    if st.session_state.selected_player_name not in player_names:
        st.session_state.selected_player_name = player_names[0]
    selected_player = st.selectbox(
        "Player", options=player_names, key="selected_player_name"
    )
    if selected_player in eligible_names:
        st.success("Forecast available for this player", icon="✅")
    else:
        st.info("Forecast not available for this player — outside the model's evaluation set", icon="ℹ️")
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

    # ---------------------------------------------------------------- 5-season history
    st.divider()
    history_df = load_player_history()
    player_history = history_df[history_df["player"] == selected_player].sort_values("season")

    if player_history.empty:
        st.info(f"No prior-season history (2019-20 to 2023-24) found for {selected_player}.")
    else:
        player_history = player_history.copy()
        player_history["season_label"] = player_history["season"].apply(format_season_label)

        st.subheader("5-season history (2019-20 to 2023-24)")
        st.plotly_chart(
            build_history_chart(player_history, selected_player), use_container_width=True
        )
        st.plotly_chart(
            build_minutes_chart(player_history, selected_player), use_container_width=True
        )

else:
    st.header("Forecast results — 2024-25 season")
    st.caption(
        "This model used five prior seasons (2019-20 to 2023-24) as input to predict "
        "the already-known 2024-25 season — a retrospective check of the model, not a live forecast."
    )
    st.info(
        "Model tested across 264 players with 5 complete prior seasons — "
        "explains 58% of the variation in next-season non-penalty goals/90 (test R² = 0.58)."
    )
    forecast_df = load_forecast_results()
    player_display_names = {pk: pk.rsplit("_", 1)[0] for pk in eligible_players}

    selected_player = st.session_state.selected_player_name
    st.subheader(f"Selected player: {selected_player}")
    st.caption("Change the player from the Scouting comparison view.")

    matching_pks = [pk for pk, name in player_display_names.items() if name == selected_player]

    if not matching_pks:
        st.info(
            f"{selected_player} is outside the model's 264-player evaluation set — "
            "pick a different player from the Scouting comparison view to see a forecast.",
            icon="ℹ️",
        )
    else:
        selected_player_key = matching_pks[0]
        player_forecast = forecast_df[forecast_df["player_key"] == selected_player_key]

        if player_forecast.empty:
            st.warning(f"No forecast result found for {selected_player}.")
        else:
            row = player_forecast.iloc[0]
            actual = row["actual_2425"]
            predicted = row["predicted_2425"]
            gap = predicted - actual

            m1, m2, m3 = st.columns(3)
            m1.metric("Actual (2024-25)", f"{actual:.2f}")
            m2.metric("Predicted (2024-25)", f"{predicted:.2f}")
            m3.metric("Gap", f"{gap:+.2f}", delta=f"{gap:+.2f}")

            st.plotly_chart(
                build_forecast_chart(selected_player, actual, predicted),
                use_container_width=True,
            )

            direction = "overshot" if gap > 0 else "undershot"
            st.caption(
                f"Using the five prior seasons (2019-20 to 2023-24) as input, the model "
                f"{direction} {selected_player}'s already-known 2024-25 non-penalty "
                f"goals/90 by {abs(gap):.2f} ({predicted:.2f} predicted vs. {actual:.2f} actual). "
                f"This is a retrospective check against a season that already happened, not a "
                f"live forecast."
            )

            st.divider()
            abs_gaps = (forecast_df["predicted_2425"] - forecast_df["actual_2425"]).abs()
            percentile = (abs_gaps <= abs(gap)).mean() * 100
            st.write(
                f"**{selected_player}'s prediction miss is smaller than "
                f"{percentile:.0f}% of all 264 players** in the study — the model "
                f"predicted them {'more' if percentile >= 50 else 'less'} accurately "
                f"than most players evaluated."
            )
            st.plotly_chart(
                build_gap_distribution_chart(forecast_df, gap, selected_player),
                use_container_width=True,
            )
            st.caption(
                "Each bar shows how many players had a miss of that size. "
                f"The dashed line marks where {selected_player} falls."
            )

            st.divider()
            st.plotly_chart(
                build_full_forecast_scatter(forecast_df, selected_player_key, selected_player),
                use_container_width=True,
            )
            st.caption(
                "Every one of the 264 evaluated players. Points on the dashed diagonal "
                "would mean a perfect prediction."
            )