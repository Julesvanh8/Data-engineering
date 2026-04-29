"""
dashboard.py — interactive Streamlit + Plotly dashboard for the lag analysis.

Usage:
    streamlit run vs-sandbox/dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parents[1]
PROCESSED   = ROOT / "data" / "processed"

MAIN_CSV    = PROCESSED / "merged_monthly.csv"
EVENTS_CSV  = PROCESSED / "events_combined.csv"
NAMED_COLORS = {
    "Dot-com crash":           "rgba(255, 180, 180, 0.35)",
    "Global Financial Crisis": "rgba(255, 200, 150, 0.35)",
    "COVID crash":             "rgba(180, 220, 255, 0.35)",
}
OTHER_COLOR = "rgba(210, 210, 210, 0.35)"

# ── data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data
def load_main() -> pd.DataFrame:
    df = pd.read_csv(MAIN_CSV, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df

@st.cache_data
def load_raw_events() -> pd.DataFrame:
    if not EVENTS_CSV.exists():
        return pd.DataFrame(columns=["name", "lag", "unemp_change", "tax_change"])
    return pd.read_csv(EVENTS_CSV)

@st.cache_data
def load_catalog() -> pd.DataFrame:
    if not EVENTS_CSV.exists():
        return pd.DataFrame(columns=["name", "sp500_start", "sp500_trough", "sp500_pct_drop"])
    df = pd.read_csv(EVENTS_CSV, parse_dates=["sp500_start", "sp500_trough",
                                               "unemp_event_start", "unemp_peak_date",
                                               "tax_event_start",   "tax_trough_date"])
    for col in ("unemp_lag_months", "tax_lag_months", "sp500_duration"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df.drop_duplicates(subset=["name"]).reset_index(drop=True)


# ── period shading helper ─────────────────────────────────────────────────────

def add_period_shapes(fig, catalog: pd.DataFrame,
                      start_col: str, end_col: str,
                      row: int = 1, col: int = 1,
                      label_events: bool = False):
    for _, r in catalog.iterrows():
        s = r.get(start_col)
        e = r.get(end_col)
        if pd.isna(s) or pd.isna(e):
            continue
        color = NAMED_COLORS.get(r["name"], OTHER_COLOR)
        label = r["name"] if (label_events and r["name"] in NAMED_COLORS) else ""
        fig.add_vrect(
            x0=s, x1=e,
            fillcolor=color, line_width=0,
            annotation_text=label, annotation_position="top left",
            annotation_font_size=9,
            row=row, col=col,
        )


# ── page: time series ─────────────────────────────────────────────────────────

def page_time_series(df: pd.DataFrame, catalog: pd.DataFrame):
    st.header("Time Series Overview")
    show_log = st.checkbox("Show log(S&P 500)", value=True)

    with st.container(border=True):
        st.caption("Shading options")
        col1, col2, col3 = st.columns(3)
        shade_sp500 = col1.checkbox("S&P 500 drops", value=True)
        shade_unemp = col2.checkbox("Unemployment rises", value=False)
        shade_tax   = col3.checkbox("Federal tax declines", value=False)

    n_rows = 3 + (1 if show_log else 0)
    row_heights = ([0.28, 0.18] if show_log else [0.30]) + [0.26, 0.26]
    subplot_titles = (
        ["S&P 500 (Shiller)", "log(S&P 500)"] if show_log else ["S&P 500 (Shiller)"]
    ) + ["Unemployment Rate (%)", "Federal Tax Receipts (bn $, monthly)"]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        vertical_spacing=0.06,
    )

    # S&P 500
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"],
        name="S&P 500 (Shiller)", line=dict(color="#1f77b4", width=1.2),
    ), row=1, col=1)
    if shade_sp500:
        add_period_shapes(fig, catalog, "sp500_start", "sp500_trough", row=1, label_events=True)

    log_row = 2
    if show_log:
        fig.add_trace(go.Scatter(
            x=df.index, y=np.log(df["close"]),
            name="log(S&P 500)", line=dict(color="#1f77b4", width=1.2, dash="dot"),
            showlegend=True,
        ), row=2, col=1)
        if shade_sp500:
            add_period_shapes(fig, catalog, "sp500_start", "sp500_trough", row=2)
    else:
        log_row = 1

    unemp_row = log_row + 1
    tax_row   = unemp_row + 1

    # Unemployment
    fig.add_trace(go.Scatter(
        x=df.index, y=df["unemployment_rate"],
        name="Unemployment rate", line=dict(color="#d62728", width=1.2),
    ), row=unemp_row, col=1)
    if shade_sp500:
        add_period_shapes(fig, catalog, "sp500_start", "sp500_trough", row=unemp_row)
    if shade_unemp:
        add_period_shapes(fig, catalog, "unemp_event_start", "unemp_peak_date", row=unemp_row)

    # Tax receipts
    fig.add_trace(go.Scatter(
        x=df.index, y=df["receipts_bn"],
        name="Federal tax receipts (bn $)", line=dict(color="#2ca02c", width=1.2),
    ), row=tax_row, col=1)
    if shade_sp500:
        add_period_shapes(fig, catalog, "sp500_start", "sp500_trough", row=tax_row)
    if shade_tax:
        add_period_shapes(fig, catalog, "tax_event_start", "tax_trough_date", row=tax_row)

    # Range slider on the bottom x-axis
    last_xaxis = f"xaxis{n_rows}" if n_rows > 1 else "xaxis"
    fig.update_layout(**{last_xaxis: dict(rangeslider=dict(visible=True, thickness=0.05))})

    fig.update_layout(
        height=160 * n_rows + 160,
        title_text="U.S. Market & Macroeconomic Indicators",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, x=0),
        margin=dict(l=60, r=30, t=60, b=80),
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=5,  label="5Y",  step="year", stepmode="backward"),
                    dict(count=10, label="10Y", step="year", stepmode="backward"),
                    dict(count=20, label="20Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                y=1.02,
            ),
        ),
    )
    st.plotly_chart(fig, width="stretch")


# ── page: research findings ───────────────────────────────────────────────────

def page_findings(raw: pd.DataFrame, catalog: pd.DataFrame) -> None:
    st.header("Research Findings")
    st.markdown(
        "**Research question:** How long after a U.S. stock market downturn do "
        "unemployment and federal tax revenues change?"
    )
    st.markdown("---")

    if "lag" not in raw.columns or raw.empty:
        st.info("Run `analyze_lags.py` to generate event study data.")
        return

    avg_u = raw.groupby("lag")["unemp_change"].mean()
    avg_t = raw.groupby("lag")["tax_change"].mean()

    peak_u_lag = int(avg_u.idxmax())
    peak_u_val = avg_u.max()

    first_pos_u = avg_u[avg_u > 0].index.min() if (avg_u > 0).any() else None
    first_neg_t = avg_t[avg_t < 0].index.min() if (avg_t < 0).any() else None

    cat_u = catalog["unemp_lag_months"].dropna()
    cat_t = catalog["tax_lag_months"].dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Downturns analysed", len(catalog))
    c2.metric("Avg unemp peak response", f"month +{peak_u_lag}")
    c3.metric("Median unemp trough (per event)", f"month +{int(cat_u.median())}" if not cat_u.empty else "—")
    c4.metric("Median tax peak (per event)",     f"month +{int(cat_t.median())}" if not cat_t.empty else "—")

    st.markdown("---")

    col_left, col_right = st.columns(2)
    lags_x = sorted(raw["lag"].unique())

    def _study_fig(col, color, ylabel, title):
        fig = go.Figure()
        for event in raw["name"].unique():
            sub = raw[raw["name"] == event].set_index("lag").reindex(lags_x)
            fig.add_trace(go.Scatter(
                x=lags_x, y=sub[col], mode="lines",
                line=dict(color="gray", width=0.7), opacity=0.3, showlegend=False,
            ))
        avg = raw.groupby("lag")[col].mean().reindex(lags_x)
        fig.add_trace(go.Scatter(
            x=lags_x, y=avg, mode="lines+markers", name="Average",
            line=dict(color=color, width=2.5), marker=dict(size=5),
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8)
        fig.update_layout(
            title=title, xaxis_title="Months after downturn start",
            yaxis_title=ylabel, height=380, showlegend=False,
            margin=dict(t=40, b=40),
        )
        return fig

    with col_left:
        st.plotly_chart(
            _study_fig("unemp_change", "#d62728", "pp change vs baseline",
                       "Unemployment rate change after downturn"),
            width="stretch",
        )
        if first_pos_u:
            st.caption(f"Average unemployment first rises above baseline at month +{int(first_pos_u)}, "
                       f"peaking at month +{peak_u_lag} (+{peak_u_val:.2f} pp).")
        st.caption(f"Per-event median: unemployment trough at month +{int(cat_u.median())}" if not cat_u.empty else "")

    with col_right:
        st.plotly_chart(
            _study_fig("tax_change", "#2ca02c", "% change vs baseline",
                       "Federal tax receipts change after downturn"),
            width="stretch",
        )
        if first_neg_t:
            st.caption(f"Average tax receipts first fall below baseline at month +{int(first_neg_t)}.")
        else:
            st.caption(
                "Average tax receipts stay above the pre-event baseline across all lags — "
                "the long-run upward trend dominates the average. "
                "See Downturn Catalog for per-event timing."
            )
        st.caption(f"Per-event median: tax peak at month +{int(cat_t.median())}" if not cat_t.empty else "")

    st.markdown("---")
    st.markdown(
        "**Methodology note:** The event study is conditional on the "
        f"{len(catalog)} identified bear markets (≥19% drawdown from ATH). "
        "See the Downturn Catalog for per-event timing."
    )


def page_event_study(raw: pd.DataFrame):
    st.header("Event Study: Impulse Response")
    st.markdown(
        "Each gray line is one detected S&P 500 downturn event. "
        "The bold colored line is the average across **selected** events."
    )

    if "lag" not in raw.columns or raw.empty:
        st.info("Run `analyze_lags.py` to generate event study data.")
        return

    indicator = st.radio("Indicator", ["Unemployment", "Tax receipts"], horizontal=True)
    col_map = {"Unemployment": "unemp_change", "Tax receipts": "tax_change"}
    col = col_map[indicator]
    ylabel = "pp change" if indicator == "Unemployment" else "% change vs baseline"

    lags       = sorted(raw["lag"].unique())
    all_events = sorted(raw["name"].unique())

    st.markdown("""
        <style>
        [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
            background-color: #5a7fa8 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    selected = st.multiselect(
        "Include in average",
        options=all_events,
        default=all_events,
        key="event_study_sel",
    )

    fig = go.Figure()
    for event in selected:
        sub = raw[raw["name"] == event].set_index("lag").reindex(lags)
        fig.add_trace(go.Scatter(
            x=lags, y=sub[col], mode="lines",
            line=dict(color="gray", width=0.7),
            opacity=0.35, showlegend=False,
            hovertemplate=f"{event}<br>Lag %{{x}}<br>{col}=%{{y:.2f}}<extra></extra>",
        ))

    if selected:
        avg = raw[raw["name"].isin(selected)].groupby("lag")[col].mean().reindex(lags)
        color = "#c44e52" if indicator == "Unemployment" else "#3a7d50"
        fig.add_trace(go.Scatter(
            x=lags, y=avg, mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=5), name=f"Average ({len(selected)} events)",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8)
    fig.update_layout(
        xaxis_title="Months after downturn start",
        yaxis_title=ylabel,
        height=450,
        hovermode="x",
        showlegend=True,
        legend=dict(orientation="h", x=0, y=-0.15),
    )
    st.plotly_chart(fig, width="stretch")

    if not selected:
        st.warning("No events selected — select at least one to see an average.")


# ── page: lag distribution ───────────────────────────────────────────────────

def page_lag_distribution(catalog: pd.DataFrame):
    st.header("Lag Distribution")
    st.markdown(
        "For each detected downturn: how many months after the S&P 500 peak did "
        "unemployment start rising and tax receipts start falling?"
    )

    if catalog.empty or "sp500_start" not in catalog.columns:
        st.warning("No catalog found — run `analyze_lags.py` first.")
        return

    indicator = st.radio("Indicator", ["Unemployment", "Tax receipts"], horizontal=True, key="lagdist_ind")
    lag_col  = "unemp_lag_months" if indicator == "Unemployment" else "tax_lag_months"
    color    = "#d62728"          if indicator == "Unemployment" else "#2ca02c"
    label    = "Months until unemployment starts rising" if indicator == "Unemployment" \
               else "Months until tax receipts start falling"

    sub = catalog[["name", "sp500_start", "sp500_pct_drop", lag_col]].dropna(subset=[lag_col]).copy()
    sub = sub.sort_values(lag_col)
    sub["sp500_start_str"] = pd.to_datetime(sub["sp500_start"]).dt.strftime("%Y-%m")

    if sub.empty:
        st.info("No events with a detected lag.")
        return

    median_lag = sub[lag_col].median()

    col1, col2, col3 = st.columns(3)
    col1.metric("Events with detected lag", len(sub))
    col2.metric(f"Median {label.split()[2]} lag", f"{int(median_lag)} months")
    col3.metric("Range", f"{int(sub[lag_col].min())}–{int(sub[lag_col].max())} months")

    # Dot plot: one dot per event
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub[lag_col],
        y=sub["sp500_start_str"] + "  " + sub["name"],
        mode="markers",
        marker=dict(color=color, size=10, line=dict(color="white", width=1)),
        hovertemplate="<b>%{y}</b><br>Lag: %{x} months<extra></extra>",
    ))
    fig.add_vline(x=median_lag, line_dash="dash", line_color="gray",
                  annotation_text=f"median {int(median_lag)}m",
                  annotation_position="top right", annotation_font_size=11)
    fig.update_layout(
        xaxis_title=label,
        height=max(300, 35 * len(sub) + 80),
        margin=dict(l=220, r=40, t=30, b=50),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, width="stretch")

    # Bar chart: lag vs S&P drop magnitude
    st.subheader("Lag vs. S&P 500 drawdown magnitude")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=sub["sp500_pct_drop"].abs(),
        y=sub[lag_col],
        mode="markers+text",
        text=sub["sp500_start_str"].str[:4],
        textposition="top center",
        marker=dict(color=color, size=9, opacity=0.8),
        hovertemplate="<b>%{text}</b><br>Drop: %{x:.1f}%<br>Lag: %{y} months<extra></extra>",
    ))
    fig2.update_layout(
        xaxis_title="S&P 500 peak-to-trough drop (%)",
        yaxis_title=label,
        height=380,
        margin=dict(t=30, b=50),
    )
    st.plotly_chart(fig2, width="stretch")

# ── page: downturn catalog ────────────────────────────────────────────────────

def page_catalog(catalog: pd.DataFrame):
    st.header("Detected Downturn Catalog")
    st.markdown(
        "All S&P 500 downturns detected automatically (≥19% drawdown from running ATH). "
        "Shaded periods for unemployment and tax are when the respective series deteriorated."
    )

    if catalog.empty:
        st.warning("No catalog found — run `analyze_lags.py` first.")
        return

    cols = ["name", "sp500_start", "sp500_trough", "sp500_pct_drop", "sp500_duration",
            "unemp_lag_months", "unemp_event_start", "unemp_peak_date",
            "tax_lag_months",   "tax_event_start",   "tax_trough_date"]
    display = catalog[[c for c in cols if c in catalog.columns]].copy()

    rename_map = {
        "name":              "Event",
        "sp500_start":       "S&P start",
        "sp500_trough":      "S&P trough",
        "sp500_pct_drop":    "Drop (%)",
        "sp500_duration":    "S&P duration (m)",
        "unemp_lag_months":  "Unemp lag (m)",
        "unemp_event_start": "Unemp start",
        "unemp_peak_date":   "Unemp peak",
        "tax_lag_months":    "Tax lag (m)",
        "tax_event_start":   "Tax start",
        "tax_trough_date":   "Tax trough",
    }
    display.columns = [rename_map[c] for c in cols if c in catalog.columns]
    display = display.sort_values("Drop (%)")

    for col in ("S&P start", "S&P trough", "Unemp start", "Unemp peak", "Tax start", "Tax trough"):
        if col in display.columns:
            display[col] = pd.to_datetime(display[col]).dt.strftime("%Y-%m")

    fmt = {"Drop (%)": "{:.1f}", "S&P duration (m)": "{:.0f}",
           "Unemp lag (m)": "{:.0f}", "Tax lag (m)": "{:.0f}"}
    st.dataframe(
        display.style.format(fmt, na_rep="—")
            .background_gradient(subset=["Drop (%)"], cmap="RdYlGn"),
        width="stretch",
        hide_index=True,
    )

    top = display.nsmallest(15, "Drop (%)")
    fig = go.Figure(go.Bar(
        x=top["Drop (%)"],
        y=top["S&P start"].astype(str).str[:7] + " " + top["Event"],
        orientation="h",
        marker_color=[
            NAMED_COLORS.get(e, OTHER_COLOR).replace("0.35", "0.8")
            for e in top["Event"]
        ],
        text=top["Drop (%)"].map("{:.1f}%".format),
        textposition="outside",
    ))
    fig.update_layout(
        title="Top 15 Largest Downturns (peak-to-trough %)",
        xaxis_title="% drop",
        height=420,
        margin=dict(l=200, r=60, t=50, b=40),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, width="stretch")


# ── page: event deep dive ─────────────────────────────────────────────────────

def page_event_deepdive(df: pd.DataFrame, catalog: pd.DataFrame) -> None:
    st.header("Event Deep Dive")
    st.markdown(
        "For each identified downturn: when did unemployment first rise and "
        "tax revenue first drop compared to the value at the downturn start?"
    )

    if catalog.empty or "sp500_start" not in catalog.columns:
        st.warning("No catalog found — run `analyze_lags.py` first.")
        return

    events = catalog.sort_values("sp500_start")["name"].tolist()
    event  = st.selectbox("Select event", events)
    row    = catalog[catalog["name"] == event].iloc[0]

    start_date   = pd.Timestamp(row["sp500_start"])
    window_start = start_date - pd.DateOffset(months=6)
    window_end   = start_date + pd.DateOffset(months=24)
    w = df[window_start:window_end]

    lag_unemp = int(row["unemp_lag_months"]) if pd.notna(row.get("unemp_lag_months")) else None
    lag_tax   = int(row["tax_lag_months"])   if pd.notna(row.get("tax_lag_months"))   else None

    first_unemp = start_date + pd.DateOffset(months=lag_unemp) if lag_unemp else None
    first_tax   = start_date + pd.DateOffset(months=lag_tax)   if lag_tax   else None

    col1, col2, col3 = st.columns(3)
    col1.metric("Peak-to-trough drop",       f"{row['sp500_pct_drop']:.1f}%")
    col2.metric("Unemployment starts rising", f"month +{lag_unemp}" if lag_unemp else "not detected")
    col3.metric("Tax revenue starts falling", f"month +{lag_tax}"   if lag_tax   else "not detected")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["S&P 500", "Unemployment Rate (%)", "Federal Tax Receipts (bn $)"],
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.07,
    )

    fig.add_trace(go.Scatter(
        x=w.index, y=w["close"],
        name="S&P 500", line=dict(color="#1f77b4", width=1.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=w.index, y=w["unemployment_rate"],
        name="Unemployment", line=dict(color="#d62728", width=1.5),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=w.index, y=w["receipts_bn"],
        name="Tax receipts", line=dict(color="#2ca02c", width=1.5),
    ), row=3, col=1)

    def vline(date: pd.Timestamp, vrow: int, color: str, dash: str = "solid", width: float = 1.5):
        fig.add_shape(type="line",
                      x0=date, x1=date, y0=0, y1=1, yref="y domain",
                      line=dict(color=color, width=width, dash=dash),
                      row=vrow, col=1)

    vline(start_date, 1, "#1f77b4", "dash", 1.5)

    if start_date in w.index:
        fig.add_annotation(
            x=start_date, y=df.loc[start_date, "close"],
            text="Downturn start", showarrow=True, arrowhead=2,
            ax=30, ay=-40, font=dict(size=10), row=1, col=1,
        )

    if first_unemp:
        vline(first_unemp, 2, "#d62728", "dash")
        fig.add_annotation(x=first_unemp, y=1, yref="y2 domain",
                           text=f"+{lag_unemp}m", showarrow=False,
                           font=dict(color="#d62728", size=10),
                           xanchor="left", yanchor="top")

    if first_tax:
        vline(first_tax, 3, "#2ca02c", "dash")
        fig.add_annotation(x=first_tax, y=1, yref="y3 domain",
                           text=f"+{lag_tax}m", showarrow=False,
                           font=dict(color="#2ca02c", size=10),
                           xanchor="left", yanchor="top")

    fig.update_layout(
        height=650,
        hovermode="x unified",
        title_text=f"{event} — Lag Analysis",
        legend=dict(orientation="h", x=0, y=-0.08),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    st.plotly_chart(fig, width="stretch")


# ── app layout ────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Market–Macro Lag Dashboard",
        page_icon="📈",
        layout="wide",
    )
    st.title("How Long After a Stock Market Downturn Do Unemployment and Tax Revenues Change?")
    st.caption(
        "Research question: measuring the lag between U.S. S&P 500 downturns and changes "
        "in unemployment and federal income tax receipts."
    )

    df      = load_main()
    catalog = load_catalog()
    raw     = load_raw_events()
    pages = {
        "Research Findings":  lambda: page_findings(raw, catalog),
        "Time Series":        lambda: page_time_series(df, catalog),
        "Event Deep Dive":    lambda: page_event_deepdive(df, catalog),
        "Downturn Catalog":   lambda: page_catalog(catalog),
        "Lag Distribution":   lambda: page_lag_distribution(catalog),
        "Event Study":        lambda: page_event_study(raw),
    }

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Page", list(pages.keys()), label_visibility="collapsed")

    pages[page]()


if __name__ == "__main__":
    main()
