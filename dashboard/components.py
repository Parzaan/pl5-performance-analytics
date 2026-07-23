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

def build_history_chart(player_history, player_name):
    """Line chart of goals/90 and shots/90 across the 5 prior seasons."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=player_history["season_label"],
        y=player_history["goals_excl_pk_p90"],
        mode="lines+markers",
        name="Goals/90 (non-pen)",
    ))
    fig.add_trace(go.Scatter(
        x=player_history["season_label"],
        y=player_history["shots_p90"],
        mode="lines+markers",
        name="Shots/90",
    ))
    fig.update_layout(
        title=f"{player_name} — output per 90, 2019-20 to 2023-24",
        xaxis_title="Season",
        yaxis_title="Per 90 minutes",
        legend_title="Metric",
    )
    return fig


def build_minutes_chart(player_history, player_name):
    """Bar chart of minutes played across the 5 prior seasons."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=player_history["season_label"],
        y=player_history["minutes"],
        name="Minutes played",
    ))
    fig.update_layout(
        title=f"{player_name} — minutes played, 2019-20 to 2023-24",
        xaxis_title="Season",
        yaxis_title="Minutes",
    )
    return fig

def build_forecast_chart(player_name: str, actual: float, predicted: float) -> go.Figure:
    """Predicted vs. actual goals/90 for 2024-25 — retrospective check, not a live forecast."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Actual", "Predicted"],
        y=[actual, predicted],
        marker_color=[PLAYER_COLOR, LEAGUE_COLOR],
        text=[f"{actual:.2f}", f"{predicted:.2f}"],
        textposition="outside",
        textfont=dict(size=16),
    ))
    fig.update_layout(
        title=f"{player_name} — 2024-25 goals/90 (non-pen): model vs. reality",
        yaxis_title="Goals/90 (non-pen)",
        yaxis_range=[0, max(actual, predicted) * 1.3],
        showlegend=False,
        height=350,
    )
    return fig

def build_gap_distribution_chart(forecast_df: pd.DataFrame, player_gap: float, player_name: str) -> go.Figure:
    """Where this player's prediction gap sits among all 264 eligible players."""
    gaps = forecast_df["predicted_2425"] - forecast_df["actual_2425"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=gaps,
        marker_color=LEAGUE_COLOR,
        opacity=0.6,
        name="All players",
    ))
    fig.add_vline(
        x=player_gap,
        line=dict(color=PLAYER_COLOR, width=3, dash="dash"),
        annotation_text=player_name,
        annotation_position="top",
    )
    fig.update_layout(
        title="This player's miss, vs. every other player's miss",
        xaxis_title="Predicted − Actual (goals/90)",
        yaxis_title="Number of players",
        showlegend=False,
        height=350,
    )
    return fig

def build_full_forecast_scatter(forecast_df: pd.DataFrame, selected_player_key: str, selected_player_name: str) -> go.Figure:
    """Predicted vs. actual goals/90 across all 264 evaluated players, with a perfect-prediction reference line."""
    others = forecast_df[forecast_df["player_key"] != selected_player_key]
    mine = forecast_df[forecast_df["player_key"] == selected_player_key]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=others["actual_2425"], y=others["predicted_2425"],
        mode="markers", name="Other players",
        marker=dict(color=FIELD_COLOR, size=8),
        text=others["player_key"],
        hovertemplate="%{text}<br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
    ))
    if not mine.empty:
        fig.add_trace(go.Scatter(
            x=mine["actual_2425"], y=mine["predicted_2425"],
            mode="markers", name=selected_player_name,
            marker=dict(color=PLAYER_COLOR, size=16, symbol="star"),
            text=mine["player_key"],
            hovertemplate="%{text}<br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
        ))

    max_val = max(forecast_df["actual_2425"].max(), forecast_df["predicted_2425"].max()) * 1.05
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", line=dict(color="gray", dash="dash"),
        name="Perfect prediction", hoverinfo="skip",
    ))

    fig.update_layout(
        title="Predicted vs. actual, all 264 evaluated players",
        xaxis_title="Actual goals/90 (2024-25)",
        yaxis_title="Predicted goals/90 (2024-25)",
        height=450,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=40, b=70),
    )
    return fig
def build_position_bias_chart(pos_summary: pd.DataFrame) -> go.Figure:
    """Mean prediction bias (predicted − actual) per position — shows over/under-prediction direction."""
    colors = [PLAYER_COLOR if v > 0 else LEAGUE_COLOR for v in pos_summary["mean_residual"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pos_summary["primary_pos"],
        y=pos_summary["mean_residual"],
        marker_color=colors,
        text=[f"{v:+.3f}" for v in pos_summary["mean_residual"]],
        textposition="outside",
    ))
    fig.add_hline(y=0, line=dict(color="gray", width=1))
    fig.update_layout(
        title="Prediction bias by position (predicted − actual)",
        xaxis_title="Position",
        yaxis_title="Mean bias (goals/90)",
        showlegend=False,
        height=350,
    )
    return fig


def build_position_mae_chart(pos_summary: pd.DataFrame) -> go.Figure:
    """Mean absolute error per position — typical size of the miss, regardless of direction."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pos_summary["primary_pos"],
        y=pos_summary["mae"],
        marker_color=LEAGUE_COLOR,
        text=[f"{v:.3f}" for v in pos_summary["mae"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Typical prediction miss by position (MAE)",
        xaxis_title="Position",
        yaxis_title="Mean absolute error (goals/90)",
        showlegend=False,
        height=350,
    )
    return fig

def build_shap_chart(top_features: list, player_name: str) -> go.Figure:
    """Horizontal bar chart of the top features pushing this player's prediction up or down."""
    labels = [f[0] for f in top_features][::-1]
    values = [f[1] for f in top_features][::-1]
    colors = [PLAYER_COLOR if v > 0 else LEAGUE_COLOR for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))
    fig.add_vline(x=0, line=dict(color="gray", width=1))
    fig.update_layout(
        title=f"{player_name} — top factors behind this prediction",
        xaxis_title="Impact on predicted goals/90",
        showlegend=False,
        height=300,
        margin=dict(l=150, r=40, t=40, b=40),
    )
    return fig


def build_global_shap_chart(top_features: list) -> go.Figure:
    """Horizontal bar chart of the features with the largest average impact across all evaluated players."""
    labels = [f[0] for f in top_features][::-1]
    values = [f[1] for f in top_features][::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=LEAGUE_COLOR,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Features with the largest average impact on predictions",
        xaxis_title="Mean |impact| on predicted goals/90",
        showlegend=False,
        height=350,
        margin=dict(l=150, r=40, t=40, b=40),
    )
    return fig