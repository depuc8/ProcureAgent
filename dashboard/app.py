"""ProcureAgent cost / latency dashboard.

    streamlit run dashboard/app.py

Source: BigQuery request_metrics when METRICS_TO_BIGQUERY + GOOGLE_CLOUD_PROJECT are set,
otherwise the local JSONL file the TelemetryPlugin writes (METRICS_JSONL_PATH).
Each chart plots one measure on one axis. Colours are the validated blue/orange categorical
pair (light and dark steps); every chart has a table view.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.metrics_data import bucketed, load_bigquery, load_jsonl, summary
from settings import get_settings

# categorical slots 1-2, light / dark steps (checked with the palette validator in both modes)
PALETTE = {"light": ("#2a78d6", "#eb6834", "#52514e", "#d9d8d3"),
           "dark": ("#3987e5", "#d95926", "#c3c2b7", "#3a3a37")}

st.set_page_config(page_title="ProcureAgent metrics", layout="wide")


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


C1, C2, INK, GRID = PALETTE[theme_mode()]


@st.cache_data(ttl=60)
def load(source: str, days: int) -> pd.DataFrame:
    s = get_settings()
    return load_bigquery(s.bq_metrics_table_id, days) if source == "BigQuery" else load_jsonl(s.metrics_jsonl_path)


def base(df: pd.DataFrame, title: str):
    return alt.Chart(df, title=alt.TitleParams(title, anchor="start", color=INK, fontSize=14, fontWeight=600))


def axes(chart, y_title: str):
    return chart.configure_axis(labelColor=INK, titleColor=INK, gridColor=GRID, domainColor=GRID, tickColor=GRID) \
                .configure_view(stroke=None).configure_legend(labelColor=INK, titleColor=INK, orient="top")


s = get_settings()
sources = ["Local JSONL"] + (["BigQuery"] if s.project and s.metrics_to_bigquery else [])
with st.sidebar:
    source = st.radio("Source", sources, index=len(sources) - 1)
    days = st.slider("Window (days, BigQuery)", 1, 90, 14)
    freq = st.selectbox("Bucket for latency percentiles", ["15min", "1h", "6h", "1D"], index=1)
    show_table = st.toggle("Show tables", value=False)

df = load(source, days)
st.title("ProcureAgent: cost and latency")
if df.empty:
    st.info("No request metrics yet. Run a request (`adk web`, the API, or `python -m scripts.benchmark`) "
            f"and refresh. Looking in: {s.metrics_jsonl_path if source != 'BigQuery' else s.bq_metrics_table_id}")
    st.stop()

k = summary(df)
cols = st.columns(6)
cols[0].metric("Requests", f"{k['requests']}", f"{k['errors']} errors" if k["errors"] else None, delta_color="inverse")
cols[1].metric("Cost / request", f"${k['mean_cost_usd']:.4f}")
cols[2].metric("Latency p50", f"{k['p50_latency_s']:.2f}s" if k["p50_latency_s"] is not None else "n/a")
cols[3].metric("Latency p95", f"{k['p95_latency_s']:.2f}s" if k["p95_latency_s"] is not None else "n/a")
cols[4].metric("Time to first token", f"{k['mean_ttft_s']:.2f}s")
cols[5].metric("Tokens / sec", f"{k['mean_tokens_per_sec']:.0f}")

tip = [alt.Tooltip("ts:T", title="time"), alt.Tooltip("request_id:N", title="request")]

cost = (base(df, "Cost per request (USD)").mark_line(color=C1, strokeWidth=2, point=alt.OverlayMarkDef(color=C1, size=60))
        .encode(x=alt.X("ts:T", title=None), y=alt.Y("cost_usd:Q", title="USD", scale=alt.Scale(zero=True)),
                tooltip=tip + [alt.Tooltip("cost_usd:Q", format=".5f"), alt.Tooltip("total_tokens:Q")]))
st.altair_chart(axes(cost, "USD"), width="stretch")

b = bucketed(df, freq)
lat = b.melt(id_vars="ts", value_vars=["p50_latency_s", "p95_latency_s"], var_name="series", value_name="seconds")
lat["series"] = lat["series"].map({"p50_latency_s": "p50", "p95_latency_s": "p95"})
colour = alt.Color("series:N", title=None, scale=alt.Scale(domain=["p50", "p95"], range=[C1, C2]))
lines = base(lat, f"End-to-end latency, p50 and p95 per {freq}").mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60)) \
    .encode(x=alt.X("ts:T", title=None), y=alt.Y("seconds:Q", title="seconds", scale=alt.Scale(zero=True)),
            color=colour, tooltip=[alt.Tooltip("ts:T"), "series:N", alt.Tooltip("seconds:Q", format=".2f")])
last = lat.sort_values("ts").groupby("series").tail(1)
labels = alt.Chart(last).mark_text(align="left", dx=8, fontWeight=600).encode(
    x="ts:T", y="seconds:Q", text="series:N", color=colour)
st.altair_chart(axes(lines + labels, "seconds"), width="stretch")

tps = (base(df.dropna(subset=["tokens_per_sec"]), "Generation throughput (output tokens / sec in the model)")
       .mark_line(color=C1, strokeWidth=2, point=alt.OverlayMarkDef(color=C1, size=60))
       .encode(x=alt.X("ts:T", title=None), y=alt.Y("tokens_per_sec:Q", title="tokens / sec", scale=alt.Scale(zero=True)),
               tooltip=tip + [alt.Tooltip("tokens_per_sec:Q", format=".1f"), alt.Tooltip("output_tokens:Q")]))
st.altair_chart(axes(tps, "tokens / sec"), width="stretch")

if show_table:
    st.subheader("Per-bucket statistics")
    st.dataframe(b, width="stretch")
    st.subheader("Requests")
    st.dataframe(df.drop(columns=["agent_breakdown"], errors="ignore").sort_values("ts", ascending=False),
                 width="stretch")
