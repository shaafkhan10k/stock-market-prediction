import streamlit as st


def stat_card(col, label: str, val: str) -> None:
    """Render a dark stat card into a Streamlit column."""
    col.markdown(
        f"""<div class="stat-card">
            <div class="stat-title">{label}</div>
            <div class="stat-value">{val}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def info_card(label: str, value: str) -> None:
    """Render a compact info card via st.markdown."""
    st.markdown(
        f"""<div class="info-card">
            <div class="info-label">{label}</div>
            <div class="info-value">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )