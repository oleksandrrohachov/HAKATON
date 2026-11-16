"""Small reusable Streamlit pieces."""
import plotly.graph_objects as go
import streamlit as st

from glossary import CHART_HELP


def chart(fig: go.Figure, help_key: str, container=st) -> None:
    """Show a plotly figure with its plain-language explanation underneath."""
    container.plotly_chart(fig, width="stretch")
    container.caption(f"ℹ️ {CHART_HELP[help_key]}")
