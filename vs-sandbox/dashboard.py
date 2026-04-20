"""
dashboard.py — interactive Streamlit + Plotly dashboard for the lag analysis.

Usage:
    streamlit run vs-sandbox/dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parents[1]
PROCESSED   = ROOT / "data" / "processed"
RAW         = ROOT / "data" / "raw"

MAIN_CSV        = PROCESSED / "merged_monthly_vs.csv"
LAG_CSV         = PROCESSED / "lag_results.csv"
EVENTS_CSV      = PROCESSED / "events_combined.csv"
TAX_SOURCES_CSV = PROCESSED / "tax_sources.csv"
FRED_UNRATE_CSV = RAW / "fred_unrate.csv"

NAMED_EVENT_NAMES = {"Dot-com crash", "Global Financial Crisis", "COVID crash"}

NAMED_COLORS = {
    "Dot-com crash":           "rgba(255, 180, 180, 0.35)",
    "Global Financial Crisis": "rgba(255, 200, 150, 0.35)",
    "COVID crash":             "rgba(180, 220, 255, 0.35)",
}
OTHER_COLOR = "rgba(210, 210, 210, 0.35)"

PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
           "#edc948", "#b07aa1", "#ff9da7", "#9c755f"]

# ── data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data
def load_main() -> pd.DataFrame:
    df = pd.read_csv(MAIN_CSV, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df

@st.cache_data
def load_lags() -> pd.DataFrame:
    return pd.read_csv(LAG_CSV)

@st.cache_data
def load_raw_events() -> pd.DataFrame:
    if not EVENTS_CSV.exists():
        return pd.DataFrame(columns=["name", "lag", "unemp_change", "tax_change"])
    return pd.read_csv(EVENTS_CSV)

@st.cache_data
def load_named_events() -> pd.DataFrame:
    if not EVENTS_CSV.exists():
        return pd.DataFrame(columns=["name", "lag", "unemp_change", "tax_change"])
    df = pd.read_csv(EVENTS_CSV)
    return df[df["name"].isin(NAMED_EVENT_NAMES)]

@st.cache_data
def load_tax_sources() -> pd.DataFrame:
    df = pd.read_csv(TAX_SOURCES_CSV, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df

@st.cache_data
def load_fred_unrate() -> pd.Series:
    df = pd.read_csv(FRED_UNRATE_CSV, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    return df["UNRATE"]

@st.cache_data
def load_catalog() -> pd.DataFrame:
    if not EVENTS_CSV.exists():
        return pd.DataFrame(columns=["name", "start_date", "trough_date", "pct_drop"])
    df = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "trough_date"])
    for col in ("unemp_lag_months", "tax_lag_months", "duration_trough_unemp", "duration_trough_tax"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df.drop_duplicates(subset=["name"]).reset_index(drop=True)


# ── downturn shading helper ───────────────────────────────────────────────────

def add_downturn_shapes(fig, catalog: pd.DataFrame, row: int = 1, col: int = 1):
    for _, r in catalog.iterrows():
        color = NAMED_COLORS.get(r["name"], OTHER_COLOR)
        label = r["name"] if r["name"] in NAMED_COLORS else ""
        fig.add_vrect(
            x0=r["start_date"], x1=r["trough_date"],
            fillcolor=color, line_width=0,
            annotation_text=label, annotation_position="top left",
            annotation_font_size=9,
            row=row, col=col,
        )


# ── page: time series ─────────────────────────────────────────────────────────

def page_time_series(df: pd.DataFrame, tax: pd.DataFrame, fred_unrate: pd.Series,
                     catalog: pd.DataFrame):
    st.header("Time Series Overview")
    show_log = st.checkbox("Show log(S&P 500)", value=True)

    df_f  = df
    tax_f = tax
    fr_f  = fred_unrate

    n_rows = 3 + (1 if show_log else 0)
    row_heights = ([0.28, 0.18] if show_log else [0.30]) + [0.26, 0.26]
    subplot_titles = (
        ["S&P 500 (Shiller)", "log(S&P 500)"] if show_log else ["S&P 500 (Shiller)"]
    ) + ["Unemployment Rate (%)", "Federal Tax Receipts (bn $)"]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        vertical_spacing=0.06,
    )

    # S&P 500
    fig.add_trace(go.Scatter(
        x=df_f.index, y=df_f["close"],
        name="S&P 500 (Shiller)", line=dict(color="#1f77b4", width=1.2),
    ), row=1, col=1)
    add_downturn_shapes(fig, catalog, row=1)

    # log(S&P)
    log_row = 2
    if show_log:
        fig.add_trace(go.Scatter(
            x=df_f.index, y=np.log(df_f["close"]),
            name="log(S&P 500)", line=dict(color="#1f77b4", width=1.2, dash="dot"),
            showlegend=True,
        ), row=2, col=1)
        add_downturn_shapes(fig, catalog, row=2)
    else:
        log_row = 1  # skip

    unemp_row = log_row + 1
    tax_row   = unemp_row + 1

    # Unemployment — FRED only
    fig.add_trace(go.Scatter(
        x=fr_f.index, y=fr_f,
        name="FRED UNRATE", line=dict(color="#d62728", width=1.2),
    ), row=unemp_row, col=1)
    add_downturn_shapes(fig, catalog, row=unemp_row)

    # Tax
    if "fred_bn" in tax_f.columns:
        fig.add_trace(go.Scatter(
            x=tax_f.index, y=tax_f["fred_bn"],
            name="FRED W006RC1Q027SBEA (SAAR÷12)", line=dict(color="#2ca02c", width=1.2),
        ), row=tax_row, col=1)
    add_downturn_shapes(fig, catalog, row=tax_row)

    fig.update_layout(
        height=160 * n_rows + 120,
        title_text="U.S. Market & Macroeconomic Indicators",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, x=0),
        margin=dict(l=60, r=30, t=60, b=60),
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


# ── page: event study ─────────────────────────────────────────────────────────

def page_findings(lags: pd.DataFrame, raw: pd.DataFrame, catalog: pd.DataFrame) -> None:
    st.header("Research Findings")
    st.markdown(
        "**Research question:** How long after a U.S. stock market downturn do "
        "unemployment and federal tax revenues change?"
    )
    st.markdown("---")

    # ── derive headline numbers from event study ──────────────────────────────
    avg_u = raw.groupby("lag")["unemp_change"].mean()
    avg_t = raw.groupby("lag")["tax_change"].mean()

    peak_u_lag = int(avg_u.idxmax())
    peak_u_val = avg_u.max()

    first_pos_u = avg_u[avg_u > 0].index.min() if (avg_u > 0).any() else None
    first_neg_t = avg_t[avg_t < 0].index.min() if (avg_t < 0).any() else None

    # median per-event trough lag from catalog
    cat_u = catalog["unemp_lag_months"].dropna()
    cat_t = catalog["tax_lag_months"].dropna()

    # ── key metrics row ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Downturns analysed", len(catalog))
    c2.metric("Avg unemp peak response", f"month +{peak_u_lag}")
    c3.metric("Median unemp trough (per event)", f"month +{int(cat_u.median())}" if not cat_u.empty else "—")
    c4.metric("Median tax peak (per event)",     f"month +{int(cat_t.median())}" if not cat_t.empty else "—")

    st.markdown("---")

    # ── side-by-side event study charts ──────────────────────────────────────
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
        "**Methodology note:** The event study is the primary analysis — it is conditional on the "
        f"{len(catalog)} identified bear markets (≥19% drawdown from ATH). "
        "The cross-correlation (see sidebar) uses all ~900 months of data and is not event-conditional; "
        "its signal is diluted by normal months but provides a larger-sample statistical validation."
    )


def page_event_study(raw: pd.DataFrame):
    st.header("Event Study: Impulse Response")
    st.markdown(
        "Each gray line is one detected S&P 500 downturn event. "
        "The bold colored line is the average across all events."
    )

    indicator = st.radio("Indicator", ["Unemployment", "Tax receipts"], horizontal=True)
    col_map = {"Unemployment": "unemp_change", "Tax receipts": "tax_change"}
    col = col_map[indicator]
    ylabel = "pp change" if indicator == "Unemployment" else "% change vs baseline"

    lags   = sorted(raw["lag"].unique())
    events = raw["name"].unique()

    fig = go.Figure()
    for event in events:
        sub = raw[raw["name"] == event].set_index("lag").reindex(lags)
        fig.add_trace(go.Scatter(
            x=lags, y=sub[col], mode="lines",
            line=dict(color="gray", width=0.7),
            opacity=0.35, showlegend=False,
            hovertemplate=f"Event {event}<br>Lag %{{x}}<br>{col}=%{{y:.2f}}<extra></extra>",
        ))

    avg = raw.groupby("lag")[col].mean().reindex(lags)
    color = "#d62728" if indicator == "Unemployment" else "#2ca02c"
    fig.add_trace(go.Scatter(
        x=lags, y=avg, mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=5), name="Average",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8)
    fig.update_layout(
        xaxis_title="Months after downturn start",
        yaxis_title=ylabel,
        height=450,
        hovermode="x",
    )
    st.plotly_chart(fig, width="stretch")


# ── page: named event lags ────────────────────────────────────────────────────

def page_named_events(named: pd.DataFrame):
    st.header("Lag Analysis per Named Downturn")
    st.markdown("Change vs. 6-month pre-event baseline for Dot-com (2000), Global Financial Crisis (2008), and COVID (2020).")

    indicator = st.radio("Indicator", ["Unemployment", "Tax receipts"], horizontal=True, key="ne_ind")
    col = "unemp_change" if indicator == "Unemployment" else "tax_change"
    ylabel = "pp change" if indicator == "Unemployment" else "% change vs baseline"

    lags = sorted(named["lag"].unique())
    fig = go.Figure()

    events = sorted(named["name"].unique(),
                    key=lambda e: named[named["name"] == e]["lag"].count(), reverse=True)
    for i, event in enumerate(events):
        color = PALETTE[i % len(PALETTE)]
        sub = named[named["name"] == event].set_index("lag").reindex(lags)
        if sub[col].notna().any():
            fig.add_trace(go.Scatter(
                x=lags, y=sub[col], mode="lines+markers",
                name=event, line=dict(color=color, width=2),
                marker=dict(size=5),
            ))

    legend_pos = dict(xanchor="right", x=1, yanchor="top", y=1) if indicator == "Unemployment" \
                 else dict(orientation="h", xanchor="left", x=0, yanchor="bottom", y=0,
                           entrywidthmode="fraction", entrywidth=0.5)

    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8)
    fig.update_layout(
        xaxis_title="Months after downturn start",
        yaxis_title=ylabel,
        height=450,
        hovermode="x unified",
        legend=legend_pos,
    )
    st.plotly_chart(fig, width="stretch")


# ── page: cross-correlation ───────────────────────────────────────────────────

def page_cross_correlation(lags: pd.DataFrame):
    st.header("Cross-Correlation Analysis")
    st.markdown(
        "Pearson correlation between S&P 500 monthly return and the **lagged** change in each "
        "economic indicator. Starred bars are significant at p < 0.05."
    )

    indicator = st.radio("Indicator", ["Unemployment", "Tax receipts"], horizontal=True, key="xcorr_ind")
    r_col = "r_unemp" if indicator == "Unemployment" else "r_tax"
    p_col = "p_unemp_xcorr" if indicator == "Unemployment" else "p_tax_xcorr"

    colors = [
        "#1f77b4" if p < 0.05 else "#aec7e8"
        for p in lags[p_col]
    ]
    stars = ["★" if p < 0.05 else "" for p in lags[p_col]]

    fig = go.Figure(go.Bar(
        x=lags["lag"], y=lags[r_col],
        marker_color=colors,
        text=stars, textposition="outside",
        hovertemplate="Lag %{x} months<br>r = %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="black", line_width=0.8)
    fig.update_layout(
        xaxis_title="Lag (months)",
        yaxis_title="Pearson r",
        xaxis=dict(tickmode="linear", tick0=lags["lag"].min(), dtick=1),
        height=420,
    )
    st.plotly_chart(fig, width="stretch")

    # Summary table
    st.subheader("Full lag table")
    show_cols = ["lag", "avg_unemp_change", "avg_tax_change",
                 "r_unemp", "p_unemp_xcorr", "r_tax", "p_tax_xcorr", "n_events"]
    avail = [c for c in show_cols if c in lags.columns]
    st.dataframe(
        lags[avail].style.format({c: "{:.4f}" for c in avail if c not in ("lag", "n_events")}),
        width="stretch",
    )


# ── page: heatmap ─────────────────────────────────────────────────────────────

def page_heatmap(lags: pd.DataFrame):
    st.header("Correlation Heatmap")

    heat = lags.set_index("lag")[["r_unemp", "r_tax"]].T
    heat.index = ["Unemployment", "Tax receipts"]

    fig = px.imshow(
        heat,
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        zmin=-0.5, zmax=0.5,
        text_auto=".3f",
        aspect="auto",
        labels=dict(x="Lag (months)", color="Pearson r"),
    )
    fig.update_layout(
        title="Cross-Correlation Heatmap: S&P 500 Return → Lagged Economic Change",
        height=260,
        coloraxis_colorbar=dict(title="r"),
        margin=dict(l=120, r=30, t=60, b=60),
    )
    st.plotly_chart(fig, width="stretch")


# ── page: downturn catalog ────────────────────────────────────────────────────

def page_catalog(catalog: pd.DataFrame):
    st.header("Detected Downturn Catalog")
    st.markdown(
        "All S&P 500 downturns detected automatically (monthly return < −5%, "
        "events within 3 months merged). Start = first drop below threshold; "
        "Trough = lowest close within 24 months after start."
    )

    if catalog.empty:
        st.warning("No catalog found — run `analyze_lags.py` first.")
        return

    cols = ["name", "start_date", "trough_date", "peak_close", "trough_close",
            "pct_drop", "duration_trough_sp500",
            "unemp_lag_months", "unemp_rise_pp", "duration_trough_unemp",
            "tax_lag_months",   "tax_drop_pct",  "duration_trough_tax"]
    display = catalog[[c for c in cols if c in catalog.columns]].copy()
    display.columns = [{"name": "Event", "start_date": "Start", "trough_date": "Trough",
                        "peak_close": "S&P at Start", "trough_close": "S&P at Trough",
                        "pct_drop": "Drop (%)",
                        "duration_trough_sp500":  "S&P trough (m)",
                        "unemp_lag_months":        "Unemp rise starts (m)",
                        "unemp_rise_pp":           "Unemp rise (pp)",
                        "duration_trough_unemp":   "Unemp peaks (m)",
                        "tax_lag_months":           "Tax drop starts (m)",
                        "tax_drop_pct":             "Tax drop (%)",
                        "duration_trough_tax":      "Tax troughs (m)"}[c]
                       for c in cols if c in catalog.columns]
    display = display.sort_values("Drop (%)")
    for col in ("Start", "Trough"):
        if col in display.columns:
            display[col] = pd.to_datetime(display[col]).dt.strftime("%Y-%m")

    fmt = {"Drop (%)": "{:.1f}", "S&P at Start": "{:.0f}", "S&P at Trough": "{:.0f}",
           "Unemp rise starts (m)": "{:.0f}", "Unemp rise (pp)": "{:.1f}", "Unemp peaks (m)": "{:.0f}",
           "Tax drop starts (m)":   "{:.0f}", "Tax drop (%)":    "{:.1f}", "Tax troughs (m)":  "{:.0f}"}
    st.dataframe(
        display.style.format(fmt, na_rep="—")
            .background_gradient(subset=["Drop (%)"], cmap="RdYlGn"),
        width="stretch",
        hide_index=True,
    )

    # Bar chart: top 15 by magnitude
    top = display.nsmallest(15, "Drop (%)")
    fig = go.Figure(go.Bar(
        x=top["Drop (%)"],
        y=top["Start"].astype(str).str[:7] + " " + top["Event"],
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

    events = catalog.sort_values("start_date")["name"].tolist()
    event  = st.selectbox("Select event", events)
    row    = catalog[catalog["name"] == event].iloc[0]

    start_date  = pd.Timestamp(row["start_date"])
    BEFORE      = 6
    AFTER       = 24

    window_start = start_date - pd.DateOffset(months=BEFORE)
    window_end   = start_date + pd.DateOffset(months=AFTER)

    # Slice to window so y-axis scales to visible data
    w = df[window_start:window_end]

    # Read lags from catalog (already computed in analyze_lags.py)
    lag_unemp = int(row["unemp_lag_months"]) if pd.notna(row.get("unemp_lag_months")) else None
    lag_tax   = int(row["tax_lag_months"])   if pd.notna(row.get("tax_lag_months"))   else None

    first_unemp = start_date + pd.DateOffset(months=lag_unemp) if lag_unemp else None
    first_tax   = start_date + pd.DateOffset(months=lag_tax)   if lag_tax   else None

    # ── summary metrics ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Peak-to-trough drop",      f"{row['pct_drop']:.1f}%")
    col2.metric("Unemployment starts rising", f"month +{lag_unemp}" if lag_unemp else "not detected")
    col3.metric("Tax revenue starts falling",  f"month +{lag_tax}"   if lag_tax   else "not detected")

    # ── three-panel chart ─────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["S&P 500", "Unemployment Rate (%)", "Federal Tax Receipts (bn $)"],
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.07,
    )

    # S&P 500
    fig.add_trace(go.Scatter(
        x=w.index, y=w["close"],
        name="S&P 500", line=dict(color="#1f77b4", width=1.5),
    ), row=1, col=1)

    # Unemployment
    fig.add_trace(go.Scatter(
        x=w.index, y=w["unemployment_rate"],
        name="Unemployment", line=dict(color="#d62728", width=1.5),
    ), row=2, col=1)

    # Tax
    fig.add_trace(go.Scatter(
        x=w.index, y=w["receipts_bn"],
        name="Tax receipts", line=dict(color="#2ca02c", width=1.5),
    ), row=3, col=1)

    def vline(date: pd.Timestamp, row: int, color: str, dash: str = "solid", width: float = 1.5):
        fig.add_shape(type="line",
                      x0=date, x1=date, y0=0, y1=1, yref="y domain",
                      line=dict(color=color, width=width, dash=dash),
                      row=row, col=1)

    # Downturn start: dashed blue line on S&P panel only
    vline(start_date, 1, "#1f77b4", "dash", 1.5)

    # Downturn start annotation on S&P panel
    if start_date in w.index:
        fig.add_annotation(
            x=start_date, y=df.loc[start_date, "close"],
            text="Downturn start", showarrow=True, arrowhead=2,
            ax=30, ay=-40, font=dict(size=10), row=1, col=1,
        )

    # Unemployment rise: dashed red line + label
    if first_unemp:
        vline(first_unemp, 2, "#d62728", "dash")
        fig.add_annotation(x=first_unemp, y=1, yref="y2 domain",
                           text=f"+{lag_unemp}m", showarrow=False,
                           font=dict(color="#d62728", size=10),
                           xanchor="left", yanchor="top")

    # Tax drop: dashed green line + label
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

    df         = load_main()
    lags       = load_lags()
    raw        = load_raw_events()
    named      = load_named_events()
    tax        = load_tax_sources()
    fred_unemp = load_fred_unrate()
    catalog    = load_catalog()

    pages = {
        "Research Findings":    lambda: page_findings(lags, raw, catalog),
        "Event Study":          lambda: page_event_study(raw),
        "Per-Event Analysis":   lambda: page_named_events(named),
        "Event Deep Dive":      lambda: page_event_deepdive(df, catalog),
        "Downturn Catalog":     lambda: page_catalog(catalog),
        "Time Series":          lambda: page_time_series(df, tax, fred_unemp, catalog),
        "Cross-Correlation":    lambda: page_cross_correlation(lags),
        "Heatmap":              lambda: page_heatmap(lags),
    }

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Page", list(pages.keys()), label_visibility="collapsed")

    pages[page]()


if __name__ == "__main__":
    main()
