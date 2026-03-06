import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# Defect Post-Analysis Dashboard
# =========================================================

st.set_page_config(page_title="Defect Post Analysis", layout="wide")

DATA_PATH = "mice_final_data_with_id.csv"

# ----------------------------
# Theme / spacing
# ----------------------------
SAMSUNG_BLUE = "#1428A0"
BG = "#F9FAFB"
PANEL = "#FFFFFF"
STROKE = "#E5E7EB"
TEXT = "#111827"
MUTED = "#6B7280"

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

st.markdown(
    f"""
    <style>
      :root {{
        --bg:{BG}; --panel:{PANEL}; --stroke:{STROKE};
        --text:{TEXT}; --muted:{MUTED}; --blue:{SAMSUNG_BLUE};
      }}

      .stApp {{ background: var(--bg); color: var(--text); }}
      header, [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
      footer {{ visibility: hidden; }}

      [data-testid="stSidebar"] {{
        background-color: #F8FAFC;
        border-right: 1px solid #E5E7EB;
      }}

      .block-container {{
        padding-top: 0.5rem;
        padding-bottom: 0.9rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        max-width: 100%;
      }}

      hr {{
        border: none;
        border-top: 1px solid var(--stroke);
        margin: 6px 0;
      }}

      .topbar {{
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 50px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      }}

      .brand {{
        font-size: 24px;
        font-weight: 900;
        letter-spacing: 0.2px;
        color: var(--blue);
      }}

      .subtitle {{
        color: var(--muted);
        font-weight: 700;
        margin-top: 2px;
        font-size: 12px;
      }}

      .metric-block {{
        height: 220px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        padding-top: 2px;
      }}

      .metric-block .title-box {{
        display: inline-block;
        padding: 20px 70px;
        background: var(--panel);
        border: 1px solid var(--stroke);
        border-radius: 12px;
        color: var(--text);
        font-size: 16px;
        font-weight: 900;
        line-height: 1.2;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        margin-bottom: 70px;
      }}

      .metric-block .metric-value-wrap {{
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
      }}

      .metric-block .metric-value {{
        font-size: 64px;
        font-weight: 950;
        color: var(--blue);
        line-height: 1;
        letter-spacing: -1px;
      }}

      .card-title {{
        display: block;
        width: fit-content;
        padding: 17px 70px;
        border: 1px solid var(--stroke);
        border-radius: 12px;
        background: var(--panel);
        color: var(--text);
        font-size: 16px;
        font-weight: 900;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        margin: 0 auto 8px auto;
      }}

      .tiny {{
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
      }}

      .modebar, .modebar-container {{ display: none !important; }}
      .js-plotly-plot .plotly .modebar {{ display: none !important; }}
      [data-testid="stPlotlyChart"] .modebar {{ display: none !important; }}

      div[data-testid="stVerticalBlock"] > div {{ gap: 0.35rem; }}
      .element-container {{ margin-bottom: 0.35rem !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

def style_fig(fig, height: int):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=34, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
    )
    fig.update_xaxes(
        gridcolor="rgba(17,24,39,0.08)",
        zerolinecolor="rgba(17,24,39,0.10)",
    )
    fig.update_yaxes(
        gridcolor="rgba(17,24,39,0.08)",
        zerolinecolor="rgba(17,24,39,0.10)",
    )
    return fig

# ----------------------------
# Load
# ----------------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

df = load_csv(DATA_PATH)

if "id" not in df.columns or "label" not in df.columns:
    st.error("CSV에 'id', 'label' 컬럼이 필요합니다.")
    st.stop()

df = df.copy()
df["_id_num"] = pd.to_numeric(df["id"], errors="coerce")
df["_id_sort"] = df["_id_num"] if df["_id_num"].notna().mean() > 0.8 else df["id"].astype(str)
df = df.sort_values("_id_sort").reset_index(drop=True)

FEATURE_COLS = [c for c in df.columns if c not in ("id", "label", "_id_num", "_id_sort")]

for c in FEATURE_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

def infer_stages_metrics(cols):
    stages = set()
    metrics = set()
    for c in cols:
        if c.startswith("stage") and "_" in c:
            left, metric = c.split("_", 1)
            if left[5:].isdigit():
                stages.add(int(left[5:]))
                metrics.add(metric)
    return sorted(stages), sorted(metrics)

STAGES, METRICS_ALL = infer_stages_metrics(FEATURE_COLS)
if not STAGES:
    st.error("stage1~ 구조를 찾지 못했습니다. (예: stage1_temp)")
    st.stop()

CORE_METRICS_ORDER = [
    "temp",
    "humidity",
    "flow_deviation",
    "density_deviation",
    "viscosity_deviation",
    "o2_deviation",
    "n_deviation",
    "co2_deviation",
]
METRICS = [m for m in CORE_METRICS_ORDER if m in METRICS_ALL]
if not METRICS:
    st.error("8개 핵심 지표(temp/humidity/... )를 찾지 못했습니다.")
    st.stop()

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
    <div class="topbar">
      <div class="brand">SAMSUNG ELECTRONICS · 불량제품 사후분석</div>
      <div class="subtitle">요약 · 선택 범위 분포 · 불량 비율 확인</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Summary
# ----------------------------
total_prod = int(len(df))
defect_count = int((df["label"] == 1).sum())
ok_count = total_prod - defect_count

# ----------------------------
# Row A: KPI + KPI + Donut
# ----------------------------
a1, a2, a3, a4 = st.columns([0.7, 0.7, 0.7, 1.0], gap="medium")

with a1:
    st.markdown(
        f"""
        <div class="metric-block">
          <div class="title-box">총 생산수 (전체)</div>
          <div class="metric-value-wrap">
            <div class="metric-value">{total_prod:,} 개</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with a2:
    st.markdown(
        f"""
        <div class="metric-block">
          <div class="title-box">불량 개수 (전체)</div>
          <div class="metric-value-wrap">
            <div class="metric-value">{defect_count:,} 개</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with a3:
    st.markdown('<div class="card-title">전체 불량 비율</div>', unsafe_allow_html=True)

    pie_df = pd.DataFrame(
        {"status": ["정상(0)", "불량(1)"], "count": [ok_count, defect_count]}
    )
    fig_pie = px.pie(pie_df, values="count", names="status", hole=0.4)
    fig_pie.update_traces(textposition="inside", textinfo="percent")
    fig_pie.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0.7,
        ),
    )
    fig_pie = style_fig(fig_pie, height=220)
    st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CONFIG)