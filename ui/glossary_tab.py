"""Tab "Словник": every term used on the site in plain language."""
import streamlit as st

from glossary import TERMS


def render() -> None:
    """Definition list of all terms."""
    st.markdown("Пояснення всіх термінів, які трапляються на сайті. Під кожним графіком також є коротка підказка ℹ️.")
    for term, meaning in TERMS.items():
        st.markdown(f"**{term}** — {meaning}")
