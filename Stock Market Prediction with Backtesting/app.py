import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import precision_score
from sklearn.calibration import calibration_curve

from src.data_loader import fetch_sp500_data, fetch_macro_indicators, DATA_DIR
from src.features import prepare_dataset
from src.backtester import get_model
from src.metrics import compute_ml_metrics, compute_financial_returns
from src.leakage_demo import run_naive_random_split, run_proper_temporal_split
from src.utils import stat_card, info_card

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="S&P 500 ML Quant & Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #080C14; color: #E2E8F0; }

.signal-buy {
    background: linear-gradient(135deg, #052E16 0%, #064E3B 60%, #065F46 100%);
    border: 2px solid #10B981; border-radius: 16px; padding: 28px 24px;
    text-align: center; box-shadow: 0 0 40px rgba(16,185,129,0.25);
}
.signal-cash {
    background: linear-gradient(135deg, #1C0A00 0%, #451A03 60%, #78350F 100%);
    border: 2px solid #F59E0B; border-radius: 16px; padding: 28px 24px;
    text-align: center; box-shadow: 0 0 40px rgba(245,158,11,0.20);
}
.signal-title { font-size: 1.6rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing: 0.02em; }
.signal-prob  { font-size: 3.6rem; font-weight: 800; margin: 12px 0 4px 0; line-height: 1; }
.signal-sub   { font-size: 0.85rem; color: #94A3B8; margin: 0; }

.info-card {
    background: linear-gradient(135deg, #111827 0%, #0F172A 100%);
    border: 1px solid #1E293B; border-radius: 12px;
    padding: 16px 20px; margin-bottom: 10px;
}
.info-label { font-size: 0.72rem; font-weight: 700; color: #64748B;
              text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.info-value { font-size: 1.05rem; font-weight: 600; color: #F1F5F9; }

.stat-card {
    background: linear-gradient(135deg, #1E2640 0%, #0F172A 100%);
    border: 1px solid #334155; border-radius: 12px; padding: 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3); margin-bottom: 12px;
}
.stat-title { color: #94A3B8; font-size: 0.8rem; font-weight: 600;
              text-transform: uppercase; letter-spacing: 0.05em; }
.stat-value { color: #F8FAFC; font-size: 1.7rem; font-weight: 700; margin-top: 4px; }
.positive { color: #10B981; }
.negative { color: #EF4444; }

.log-header { font-size: 1.05rem; font-weight: 700; color: #E2E8F0;
              margin: 24px 0 8px 0; border-bottom: 1px solid #1E293B; padding-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA & CACHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_all_data():
    return fetch_sp500_data(), fetch_macro_indicators()


@st.cache_data(show_spinner=False)
def run_cached_walk_forward(feature_set, start_window, step_days, model_name):
    cache_file = os.path.join(
        DATA_DIR,
        f"precomputed_{feature_set}_{model_name.replace(' ', '')}_{start_window}_{step_days}.csv",
    )
    imp_file = os.path.join(
        DATA_DIR,
        f"precomputed_imp_{feature_set}_{model_name.replace(' ', '')}_{start_window}_{step_days}.csv",
    )

    sp500_df, macro_df = load_all_data()
    clean_df, predictors = prepare_dataset(sp500_df, macro_df, feature_set=feature_set)

    if os.path.exists(cache_file) and os.path.exists(imp_file):
        results_base = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        feat_imp     = pd.read_csv(imp_file, index_col=0)["Importance"]
        return results_base, feat_imp, clean_df, predictors

    all_preds, importances = [], []
    for i in range(int(start_window), clean_df.shape[0], int(step_days)):
        train = clean_df.iloc[0:i]
        test  = clean_df.iloc[i:(i + int(step_days))]
        if test.empty:
            break
        model = get_model(model_name)
        model.fit(train[predictors], train["Target"])
        probs = model.predict_proba(test[predictors])[:, 1]
        all_preds.append(
            pd.DataFrame(
                {"Target": test["Target"], "Probabilities": probs, "Close": test["Close"]},
                index=test.index,
            )
        )
        if hasattr(model, "feature_importances_"):
            importances.append(model.feature_importances_)

    results_base = pd.concat(all_preds)
    feat_imp = (
        pd.Series(np.mean(importances, axis=0), index=predictors).sort_values(ascending=False)
        if importances
        else pd.Series(dtype=float)
    )
    results_base.to_csv(cache_file)
    pd.DataFrame({"Importance": feat_imp}).to_csv(imp_file)
    return results_base, feat_imp, clean_df, predictors


@st.cache_data(ttl=300)
def get_live_signal(predictors_tuple, feature_set, start_window, step_days,
                    threshold, model_name, data_mtime):
    """Train final model on all data and produce today's signal.
    Cached until sp500.csv changes (data_mtime) or 5 min elapses."""
    sp500_df, macro_df = load_all_data()
    clean_df, predictors = prepare_dataset(sp500_df, macro_df, feature_set=feature_set)
    last_row = clean_df[predictors].iloc[-1:]
    model = get_model(model_name)
    model.fit(clean_df[predictors].iloc[:-1], clean_df["Target"].iloc[:-1])
    prob = model.predict_proba(last_row)[0][1]
    pred = int(prob >= threshold)
    return prob, pred, clean_df.index[-1], clean_df["Close"].iloc[-1], len(predictors)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Strategy Parameters")

model_name = st.sidebar.selectbox(
    "Model",
    options=["Random Forest", "XGBoost", "LightGBM"],
)
feature_set = st.sidebar.selectbox(
    "Feature Set",
    options=["all", "rolling", "baseline"],
    format_func=lambda x: {
        "all": "Full Macro + Rolling",
        "rolling": "Rolling Only",
        "baseline": "Baseline Only",
    }[x],
)
custom_threshold   = st.sidebar.slider("Confidence Threshold", 0.50, 0.70, 0.55, 0.01)
transaction_cost   = st.sidebar.selectbox(
    "Transaction Cost (bps per trade)", options=[0, 5, 10, 20], index=1
)
start_window = st.sidebar.number_input("Initial Train Window (Days)", 1000, 5000, 2500, 250)
step_days    = st.sidebar.number_input("Walk-Forward Step (Days)", 50, 500, 250, 50)
chart_days   = st.sidebar.selectbox(
    "Live Chart Range",
    [90, 180, 365, 730, 1825],
    index=2,
    format_func=lambda x: {
        90: "3 Months", 180: "6 Months", 365: "1 Year",
        730: "2 Years", 1825: "5 Years",
    }[x],
)

st.sidebar.markdown("---")
st.sidebar.info("Anti-Leakage: All features use shift(1) — no future data ever enters training.")
st.sidebar.markdown("---")
st.sidebar.markdown("v1.1.0 · Data via yfinance · MIT License")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & RUN BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading backtest results..."):
    results_base, feat_imp, clean_df, predictors = run_cached_walk_forward(
        feature_set=feature_set,
        start_window=start_window,
        step_days=step_days,
        model_name=model_name,
    )

results_df = results_base.copy()
results_df["Predictions"] = (results_df["Probabilities"] >= custom_threshold).astype(int)
ml_metrics              = compute_ml_metrics(results_df)
equity_df, fin_metrics  = compute_financial_returns(results_df, transaction_cost_bps=transaction_cost)

# Live signal — only retrain when sp500.csv actually changes
sp500_path  = os.path.join(DATA_DIR, "sp500.csv")
data_mtime  = os.path.getmtime(sp500_path) if os.path.exists(sp500_path) else 0

with st.spinner("Fetching live signal..."):
    prob, pred, latest_date, latest_close, n_feats = get_live_signal(
        tuple(predictors), feature_set, start_window, step_days,
        custom_threshold, model_name, data_mtime,
    )

sp500_full, _ = load_all_data()
chart_slice   = sp500_full.tail(chart_days)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## S&P 500 Machine Learning Predictor")
st.markdown(
    f"*Walk-Forward Backtested **{model_name}** · No Lookahead Bias · "
    f"Live Signal · {transaction_cost} bps transaction cost*"
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Live Chart & Prediction",
    "📊 Backtest & Equity Curve",
    "🔍 Feature Importance",
    "🚨 Data Leakage Audit",
    "🌐 Regime Analysis",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1  ─  LIVE CHART + SIGNAL + MARKET DETAILS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    left_col, right_col = st.columns([3, 1], gap="large")

    with left_col:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_slice.index, y=chart_slice["Close"],
            mode="lines", name="S&P 500 Close",
            line=dict(color="#38BDF8", width=2),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.07)",
        ))

        bt_chart = results_df[results_df.index >= chart_slice.index[0]].copy()
        buys  = bt_chart[bt_chart["Predictions"] == 1]
        sells = bt_chart[bt_chart["Predictions"] == 0]

        fig.add_trace(go.Scatter(
            x=buys.index, y=buys["Close"], mode="markers", name="Model: BUY",
            marker=dict(color="#10B981", size=6, symbol="triangle-up", opacity=0.85),
        ))
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells["Close"], mode="markers", name="Model: FLAT",
            marker=dict(color="#F59E0B", size=4, symbol="circle", opacity=0.35),
        ))

        vline_color = "#10B981" if pred == 1 else "#EF4444"
        fig.add_shape(
            type="line", x0=latest_date, x1=latest_date, y0=0, y1=1,
            xref="x", yref="paper", line=dict(color=vline_color, width=2, dash="dash"),
        )
        fig.add_annotation(
            x=latest_date, y=1, xref="x", yref="paper",
            text=f"  Today: {'BUY' if pred == 1 else 'FLAT'}",
            showarrow=False, font=dict(color=vline_color, size=12), xanchor="left",
        )
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#080C14", plot_bgcolor="#0D1117",
            height=480, margin=dict(l=0, r=0, t=40, b=0),
            title=dict(
                text=f"S&P 500 Price  ·  Last {chart_days} Trading Days  ·  "
                     f"Current: <b>${latest_close:,.2f}</b>",
                font=dict(size=15, color="#E2E8F0"),
            ),
            xaxis=dict(gridcolor="#1E293B"), yaxis=dict(gridcolor="#1E293B"),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                        font=dict(size=11)),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        vol_fig = go.Figure(go.Bar(
            x=chart_slice.index, y=chart_slice["Volume"],
            marker_color=np.where(chart_slice["Close"].pct_change() >= 0, "#10B981", "#EF4444"),
            opacity=0.6, name="Volume",
        ))
        vol_fig.update_layout(
            template="plotly_dark", paper_bgcolor="#080C14", plot_bgcolor="#0D1117",
            height=120, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
            xaxis=dict(showticklabels=False, gridcolor="#1E293B"),
            yaxis=dict(showticklabels=False, gridcolor="#1E293B"),
        )
        st.plotly_chart(vol_fig, use_container_width=True)

    with right_col:
        if pred == 1:
            st.markdown(f"""
            <div class="signal-buy">
                <p class="signal-title" style="color:#10B981;">BUY SIGNAL</p>
                <p class="signal-prob" style="color:#F0FDF4;">{prob:.1%}</p>
                <p class="signal-sub">Model confidence in UP move<br>Threshold: {custom_threshold:.0%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="signal-cash">
                <p class="signal-title" style="color:#F59E0B;">FLAT / CASH</p>
                <p class="signal-prob" style="color:#FFFBEB;">{prob:.1%}</p>
                <p class="signal-sub">Below confidence threshold<br>Threshold: {custom_threshold:.0%}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        info_card("Market Date",      str(latest_date.date() if hasattr(latest_date, "date") else latest_date))
        info_card("S&P 500 Close",    f"${latest_close:,.2f}")
        info_card("Model",            model_name)
        info_card("Feature Set",      f"{feature_set.upper()} · {n_feats} indicators")
        info_card("Active Threshold", f"{custom_threshold:.2f}")
        info_card("ML Precision",     f"{ml_metrics.get('Precision', 0):.2%}")
        info_card("Trade Frequency",  f"{ml_metrics.get('Trade_Frequency', 0):.1%} of days")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='info-label'>30-YEAR BACKTEST SUMMARY</div>", unsafe_allow_html=True)
        s_ret    = fin_metrics.get("Strategy_Total_Return", 0)
        b_ret    = fin_metrics.get("Benchmark_Total_Return", 0)
        s_sharpe = fin_metrics.get("Strategy_Sharpe", 0)
        col_a, col_b = st.columns(2)
        col_a.metric("Strategy", f"{s_ret:+.0%}", f"vs S&P {b_ret:+.0%}")
        col_b.metric("Sharpe",   f"{s_sharpe:.2f}", f"vs {fin_metrics.get('Benchmark_Sharpe', 0):.2f}")

    # ── Colour-coded predictions log ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Historical Walk-Forward Predictions Log")
    st.caption("Every row is an out-of-sample prediction the model made WITHOUT seeing future data.")

    display_log = results_df.copy()
    display_log["Signal"]   = np.where(display_log["Predictions"] == 1, "BUY", "FLAT")
    display_log["Outcome"]  = np.where(display_log["Target"] == 1, "UP", "DOWN")
    display_log["Correct?"] = np.where(
        display_log["Predictions"] == display_log["Target"], "Match", "Miss"
    )
    display_log["Prob (%)"] = (display_log["Probabilities"] * 100).round(1)

    log_view = display_log[["Close", "Prob (%)", "Signal", "Outcome", "Correct?"]].sort_index(
        ascending=False
    )

    def _colour_row(row):
        colour = "#052E16" if row["Correct?"] == "Match" else "#1C0A00"
        return [f"background-color: {colour}"] * len(row)

st.dataframe(
        log_view.style.apply(_colour_row, axis=1),
        use_container_width=True,
        height=350,
    )

st.download_button(
        label="⬇️ Download Predictions Log (CSV)",
        data=log_view.to_csv().encode("utf-8"),
        file_name="walk_forward_predictions_log.csv",
        mime="text/csv",
    )
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2  ─  BACKTEST & EQUITY CURVE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Strategy vs S&P 500 Buy & Hold (30-Year Walk-Forward Backtest)")

    # ── Stat cards (6 columns) ───────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    stat_card(c1, "ML Precision",    f"{ml_metrics.get('Precision', 0):.1%}")
    stat_card(c2, "Strategy Return", f"{fin_metrics.get('Strategy_Total_Return', 0):+.1%}")
    stat_card(c3, "Strategy CAGR",   f"{fin_metrics.get('Strategy_CAGR', 0):.1%}")
    stat_card(c4, "Sharpe Ratio",    f"{fin_metrics.get('Strategy_Sharpe', 0):.2f}")
    stat_card(c5, "Max Drawdown",    f"{fin_metrics.get('Strategy_Max_Drawdown', 0):.1%}")
    stat_card(c6, "Calmar Ratio",    f"{fin_metrics.get('Strategy_Calmar', 0):.2f}")

    # ── Equity curve ─────────────────────────────────────────────────────────
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["Strategy_Equity"],
        mode="lines", name=f"{model_name} Strategy (T={custom_threshold})",
        line=dict(color="#10B981", width=2.5),
    ))
    eq_fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["Benchmark_Equity"],
        mode="lines", name="S&P 500 Buy & Hold",
        line=dict(color="#38BDF8", width=1.8, dash="dash"),
    ))
    eq_fig.update_layout(
        template="plotly_dark", paper_bgcolor="#080C14", plot_bgcolor="#0D1117",
        height=420, margin=dict(l=0, r=0, t=40, b=0),
        title="Cumulative Growth of $1.00 Investment",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    st.plotly_chart(eq_fig, use_container_width=True)

    # ── Drawdown chart ───────────────────────────────────────────────────────
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["Strategy_Drawdown"] * 100,
        mode="lines", fill="tozeroy",
        fillcolor="rgba(239,68,68,0.25)", line=dict(color="#EF4444", width=1.2),
        name="Strategy Drawdown",
    ))
    dd_fig.update_layout(
        template="plotly_dark", paper_bgcolor="#080C14", plot_bgcolor="#0D1117",
        height=200, margin=dict(l=0, r=0, t=30, b=0),
        title="Strategy Drawdown (%)",
        yaxis=dict(ticksuffix="%", gridcolor="#1E293B"),
        xaxis=dict(gridcolor="#1E293B"),
        showlegend=False,
    )
    st.plotly_chart(dd_fig, use_container_width=True)

    # ── Rolling quarterly precision ──────────────────────────────────────────
    st.markdown("#### Quarterly Rolling Precision — Is the Model Stable Over Time?")
    quarterly = (
        results_df.resample("QE")
        .apply(lambda g: precision_score(g["Target"], g["Predictions"], zero_division=0))
        .rename("Precision")
    )
    rp_fig = go.Figure()
    rp_fig.add_trace(go.Scatter(
        x=quarterly.index, y=quarterly.values,
        mode="lines+markers", name="Quarterly Precision",
        line=dict(color="#10B981", width=2),
        marker=dict(size=5),
    ))
    rp_fig.add_hline(
        y=0.50, line_dash="dash", line_color="#EF4444",
        annotation_text="Random baseline (0.50)",
        annotation_position="bottom right",
    )
    rp_fig.update_layout(
        template="plotly_dark", paper_bgcolor="#080C14", plot_bgcolor="#0D1117",
        height=300, margin=dict(l=0, r=0, t=30, b=0),
        yaxis=dict(tickformat=".0%", gridcolor="#1E293B", range=[0.3, 0.8]),
        xaxis=dict(gridcolor="#1E293B"),
        hovermode="x unified",
    )
    st.plotly_chart(rp_fig, use_container_width=True)
    st.caption("A flat line = stable model. A declining trend = model decay over time.")

    # ── Annual returns table ─────────────────────────────────────────────────
    st.markdown("#### Annual Returns Breakdown")
    yearly_strat = equity_df["Strategy_Equity"].resample("YE").last().pct_change().dropna()
    yearly_bench = equity_df["Benchmark_Equity"].resample("YE").last().pct_change().dropna()
    yearly_df = pd.DataFrame({
        "Year":             yearly_strat.index.year,
        "Strategy Return":  yearly_strat.values,
        "S&P 500 Return":   yearly_bench.values,
    }).set_index("Year")
    yearly_df["Alpha"] = yearly_df["Strategy Return"] - yearly_df["S&P 500 Return"]

    def _colour_alpha(val):
        colour = "#10B981" if val >= 0 else "#EF4444"
        return f"color: {colour}; font-weight: 600"

    st.dataframe(
        yearly_df.style
        .format("{:.2%}")
        .applymap(_colour_alpha, subset=["Alpha"]),
        use_container_width=True,
        height=350,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3  ─  FEATURE IMPORTANCE + CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Which Features Move the Needle?")
    if not feat_imp.empty:
        df_imp = pd.DataFrame({"Feature": feat_imp.index, "Importance": feat_imp.values})
        fig_imp = px.bar(
            df_imp.head(15), x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Viridis",
            title="Top 15 Feature Importances (Averaged Across All Walk-Forward Folds)",
        )
        fig_imp.update_layout(
            template="plotly_dark", paper_bgcolor="#080C14", plot_bgcolor="#0D1117",
            yaxis=dict(autorange="reversed"), height=500,
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info(f"{model_name} does not expose feature importances (e.g. some LightGBM configs). Switch to Random Forest to see this chart.")

    # ── Calibration curve ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Probability Calibration Curve — Are Confidence Scores Trustworthy?")

    try:
        fraction_of_positives, mean_predicted = calibration_curve(
            results_df["Target"], results_df["Probabilities"], n_bins=10
        )
        cal_fig = go.Figure()
        cal_fig.add_trace(go.Scatter(
            x=mean_predicted, y=fraction_of_positives,
            mode="lines+markers", name=f"{model_name} Calibration",
            line=dict(color="#10B981", width=2.5),
            marker=dict(size=8),
        ))
        cal_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", name="Perfect Calibration",
            line=dict(color="#94A3B8", width=1.5, dash="dash"),
        ))
        cal_fig.update_layout(
            template="plotly_dark", paper_bgcolor="#080C14", plot_bgcolor="#0D1117",
            height=380, margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(title="Mean Predicted Probability", gridcolor="#1E293B", range=[0, 1]),
            yaxis=dict(title="Fraction of Positives (Actual Win Rate)", gridcolor="#1E293B", range=[0, 1]),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        )
        st.plotly_chart(cal_fig, use_container_width=True)
        st.caption(
            "Points **above** the diagonal → model underestimates its confidence (conservative). "
            "Points **below** → overconfident. Random Forest is typically overconfident at high probabilities."
        )
    except Exception as e:
        st.warning(f"Calibration curve unavailable: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4  ─  DATA LEAKAGE AUDIT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Data Leakage: Why Most Beginner Predictions Are Fake")
    st.markdown("""
    Standard `train_test_split(shuffle=True)` gives ~80% accuracy on stock data —
    **not** because the model is good, but because it saw the future during training.
    This button proves it:
    """)
    if st.button("Run Leakage Comparison Experiment"):
        with st.spinner("Running both methods..."):
            naive  = run_naive_random_split(clean_df, predictors)
            proper = run_proper_temporal_split(clean_df, predictors)
        col_n, col_p = st.columns(2)
        with col_n:
            st.error("NAIVE RANDOM SHUFFLE (FLAWED)")
            st.metric("Accuracy",  f"{naive['Accuracy']:.2%}")
            st.metric("Precision", f"{naive['Precision']:.2%}")
            st.warning("This looks great but FAILS in live trading — it peeked at the future.")
        with col_p:
            st.success("WALK-FORWARD TEMPORAL (RIGOROUS)")
            st.metric("Accuracy",  f"{proper['Accuracy']:.2%}")
            st.metric("Precision", f"{proper['Precision']:.2%}")
            st.info("Lower numbers, but these are REAL — the model never saw tomorrow's price.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5  ─  REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Regime Analysis — How Does the Model Behave Across Market Conditions?")

    if "VIX_Level" in clean_df.columns:
        # Merge VIX into results
        regime_df = results_df.copy()
        regime_df = regime_df.join(clean_df[["VIX_Level"]], how="left")
        regime_df["VIX_Level"] = regime_df["VIX_Level"].ffill()

        def _classify_regime(vix):
            if pd.isna(vix):
                return "Unknown"
            if vix < 15:
                return "Calm (VIX < 15)"
            elif vix < 25:
                return "Normal (15 ≤ VIX < 25)"
            else:
                return "Volatile (VIX ≥ 25)"

        regime_df["Regime"] = regime_df["VIX_Level"].apply(_classify_regime)

        regime_order = ["Calm (VIX < 15)", "Normal (15 ≤ VIX < 25)", "Volatile (VIX ≥ 25)"]
        regime_rows  = []

        for regime in regime_order:
            subset = regime_df[regime_df["Regime"] == regime]
            if len(subset) < 10:
                continue
            r_ml  = compute_ml_metrics(subset)
            _, r_fin = compute_financial_returns(subset, transaction_cost_bps=transaction_cost)
            regime_rows.append({
                "Regime":       regime,
                "Num Days":     len(subset),
                "Precision":    f"{r_ml.get('Precision', 0):.2%}",
                "Trade Freq":   f"{r_ml.get('Trade_Frequency', 0):.1%}",
                "Sharpe":       f"{r_fin.get('Strategy_Sharpe', 0):.2f}",
                "CAGR":         f"{r_fin.get('Strategy_CAGR', 0):.2%}",
                "Max Drawdown": f"{r_fin.get('Strategy_Max_Drawdown', 0):.2%}",
            })

        if regime_rows:
            st.dataframe(pd.DataFrame(regime_rows).set_index("Regime"), use_container_width=True)

            # Grouped bar chart: Precision & Sharpe by regime
            regime_chart_df = pd.DataFrame([
                {
                    "Regime":    r["Regime"],
                    "Precision": float(r["Precision"].strip("%")) / 100,
                    "Sharpe":    float(r["Sharpe"]),
                }
                for r in regime_rows
            ])

            fig_reg = go.Figure()
            fig_reg.add_trace(go.Bar(
                name="Precision", x=regime_chart_df["Regime"],
                y=regime_chart_df["Precision"],
                marker_color="#10B981", yaxis="y",
            ))
            fig_reg.add_trace(go.Bar(
                name="Sharpe", x=regime_chart_df["Regime"],
                y=regime_chart_df["Sharpe"],
                marker_color="#38BDF8", yaxis="y2",
            ))
            fig_reg.update_layout(
                template="plotly_dark", paper_bgcolor="#080C14", plot_bgcolor="#0D1117",
                height=360, margin=dict(l=0, r=60, t=40, b=0),
                barmode="group",
                title="Precision & Sharpe Ratio by Market Regime",
                yaxis=dict(
                    title="Precision", tickformat=".0%",
                    gridcolor="#1E293B", side="left",
                ),
                yaxis2=dict(
                    title="Sharpe Ratio", overlaying="y",
                    side="right", showgrid=False,
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            )
            st.plotly_chart(fig_reg, use_container_width=True)
            st.caption(
                "VIX ≥ 25 regimes are where strategies either fail catastrophically or "
                "outperform — high volatility amplifies both edge and noise."
            )
        else:
            st.warning("Not enough data per regime to compute statistics.")
    else:
        st.info(
            "VIX_Level feature not found in dataset. "
            "Switch Feature Set to **Full Macro + Rolling** to enable regime analysis."
        )