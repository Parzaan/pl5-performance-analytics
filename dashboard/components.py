"""Radar-chart and shot-quality-scatter builders, using Plotly for hover/interactivity."""

import pandas as pd
import plotly.graph_objects as go

from utils import RADAR_FEATURES, player_vector

PLAYER_COLOR = "#e34948"
LEAGUE_COLOR = "#2a78d6"
FIELD_COLOR = "rgba(42, 120, 214, 0.45)"


def build_radar(player_row: pd.Series, maxima: dict, league_avg_vector: list) -> go.Figure:
    """Radar chart: selected player vs. the (filtered) population average."""
    labels = list(RADAR_FEATURES.values())
    player_vec = player_vector(player_row, maxima)

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=league_avg_vector + league_avg_vector[:1],
            theta=labels + labels[:1],
            name="Group average",
            line=dict(color=LEAGUE_COLOR, width=2),
            fill="toself",
            fillcolor="rgba(42, 120, 214, 0.08)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=player_vec + player_vec[:1],
            theta=labels + labels[:1],
            name=str(player_row["player"]),
            line=dict(color=PLAYER_COLOR, width=2),
            fill="toself",
            fillcolor="rgba(227, 73, 72, 0.15)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=20, b=60),
        height=400,
    )
    return fig


def build_scatter(
    df: pd.DataFrame,
    selected_player: str,
    x_col: str = "shots_p90",
    y_col: str = "goals_per_shot",
) -> go.Figure:
    """Shots/90 vs finishing-rate scatter, with the selected player highlighted."""
    others = df[df["player"] != selected_player]
    mine = df[df["player"] == selected_player].dropna(subset=[x_col, y_col])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=others[x_col],
            y=others[y_col],
            mode="markers",
            name="Other players",
            marker=dict(color=FIELD_COLOR, size=8),
            text=others["player"],
            hovertemplate="%{text}<br>Sh/90: %{x:.2f}<br>G/Sh: %{y:.2f}<extra></extra>",
        )
    )
    if not mine.empty:
        fig.add_trace(
            go.Scatter(
                x=mine[x_col],
                y=mine[y_col],
                mode="markers",
                name=selected_player,
                marker=dict(color=PLAYER_COLOR, size=16, symbol="star"),
                text=mine["player"],
                hovertemplate="%{text}<br>Sh/90: %{x:.2f}<br>G/Sh: %{y:.2f}<extra></extra>",
            )
        )
    # else: player has 0 shots this season (G/Sh undefined) -- nothing to highlight.
    # The caller (app.py) shows a message above the chart in that case instead of
    # an in-plot annotation, since annotation boxes clip inconsistently on export.

    fig.update_layout(
        xaxis_title="Shots/90 (volume)",
        yaxis_title="G/Sh (finishing rate)",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=20, b=70),
        height=400,
    )
    return fig