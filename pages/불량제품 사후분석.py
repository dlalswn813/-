import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scipy_stats
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT_DIR = HERE.parents[1] if HERE.parent.name == "pages" else HERE.parent

DATA_PATH  = str(ROOT_DIR / "mice_final_data_with_id.csv")

# =========================================================
# Defect Post-Analysis Dashboard
# =========================================================

st.set_page_config(page_title="불량제품 사후분석", layout="wide")

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

      .violin-header-box {{
        display: flex;
        align-items: center;
        padding: 18px 25px;
        background: {PANEL};
        border: 1px solid {STROKE};
        /* 선(border-left)을 아예 삭제했습니다 */
        border-radius: 12px; 
        margin: 35px 0 15px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      }}

      .violin-header-box .title-text {{
        font-size: 20px;
        font-weight: 900;
        color: {TEXT};
        letter-spacing: -0.5px;
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
# 변수 한글 매핑 딕셔너리
# ----------------------------
def get_kor_label(var_name):
    name_map = {
        "temp": "온도", "humidity": "습도", "flow": "유량", 
        "density": "밀도", "viscosity": "점도", "o2": "O2", 
        "n": "N", "co2": "CO2", "deviation": "편차"
    }
    # stageX_ 제외 후 단어별로 한글 변환
    parts = var_name.split('_')[1:]
    kor_parts = [name_map.get(p, p) for p in parts]
    return " ".join(kor_parts)

# ----------------------------
# 2. 데이터 분석 로직 (두 가지 랭킹 모두 반환)
# ----------------------------
@st.cache_data
def get_analysis_data(stage_num):
    df = pd.read_csv("mice_final_data_with_id.csv")
    
    prefix = f"stage{stage_num}"
    stage_cols = [c for c in df.columns if c.startswith(prefix)]
    
    t_list = []
    for col in stage_cols:
        g0 = df[df['label'] == 0][col].dropna()
        g1 = df[df['label'] == 1][col].dropna()
        
        if len(g0) > 1 and len(g1) > 1:
            t_val, p_val = scipy_stats.ttest_ind(g0, g1)
            t_list.append({
                'var': col, 
                't_score': abs(t_val), 
                'p_value': p_val
            })
    
    # 1. 전체 순위 (바이올린 그래프 8개 구성을 위해 사용)
    t_rank = pd.DataFrame(t_list).sort_values(by='t_score', ascending=False)
    
    # 2. 유의미한 순위 (가로 막대 그래프 전용: P <= 0.05)
    significant_rank = t_rank[t_rank['p_value'] <= 0.05].copy()
    
    return df, t_rank, significant_rank


# ----------------------------
# Sidebar (Stage 필터만 유지)
# ----------------------------
st.sidebar.markdown("## 공정 필터")
selected_stage = st.sidebar.selectbox("분석 스테이지 선택", range(1, 6), index=2)

# 함수 호출하여 데이터와 분석 랭킹 가져오기
df, t_rank, significant_rank = get_analysis_data(selected_stage)

# 상위 8개 핵심 변수 리스트 추출
top_8 = t_rank.head(8)['var'].tolist()

# ----------------------------
# Header
# ----------------------------
hL, hR = st.columns([2.2, 1.0], gap="medium")
with hL:
    st.markdown(
    """
    <div class="topbar">
      <div class="brand">SPARTA ELECTRONICS · 불량제품 사후분석</div>
      <div class="subtitle">요약 · 선택 범위 분포 · 불량 비율 확인</div>
    </div>
    """,
    unsafe_allow_html=True,
)
with hR:
    st.markdown(
    f"""
    <div class="topbar">
      <div class="subtitle">현재 분석 stage</div>
      <div class="brand">stage {selected_stage} </div>
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
          <div class="title-box">전체 생산수</div>
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
          <div class="title-box">전체 불량 개수</div>
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

with a4:
    # 1. 디자인 통일 제목 (가운데 정렬 보장)
    st.markdown(f'<div class="card-title">stage{selected_stage} 품질 영향 지표</div>', unsafe_allow_html=True)
    
    if not significant_rank.empty:
        sig_plot = significant_rank.head(10).copy()
        sig_plot['kor_name'] = sig_plot['var'].apply(get_kor_label)
        
        # 2. 그래프 생성
        fig_drv = px.bar(
            sig_plot, 
            x="t_score", 
            y="kor_name", 
            orientation="h"
        )
        
        fig_drv.update_layout(
            height=250, 
            margin=dict(t=20, b=30, l=5, r=10),
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, size=11),
            
            xaxis=dict(
                title="영향도 (t-score)", 
                title_font=dict(size=10, color=MUTED),
                showgrid=True,
                gridcolor=STROKE,
                tickfont=dict(size=9, color=MUTED),
                zeroline=True,
                zerolinecolor=STROKE
            ),
            yaxis=dict(
                title=None, 
                autorange="reversed", 
                tickfont=dict(size=11, color=TEXT), 
                showgrid=False
            )
        )
        
        # 3. 색상 변경: #0068C9 적용
        fig_drv.update_traces(
            marker_color="#0068C9", 
            width=0.6
        )
        
        st.plotly_chart(fig_drv, use_container_width=True, config={"displayModeBar": False})

# ----------------------------
# 하단 상세 분석 섹션 (p-value 기반 강조 추가)
# ----------------------------
if top_8:
    
    st.markdown(
    f"""
    <div class="violin-header-box">
        <div class="title-text">
            Stage {selected_stage} 공정 지표 정밀 분석
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)
    
    # 격자선을 더 잘 보이게 하기 위한 색상 정의 (기존 STROKE보다 진한 색)
    GRID_VISIBLE = "#D1D5DB" # 더 선명한 회색
    
    groups = [top_8[:4], top_8[4:8]]
    for group in groups:
        row = st.columns(8)
        
        # 1. 바이올린 플롯 영역
        for i, var in enumerate(group):
            kor_name = get_kor_label(var)
            p_val = t_rank[t_rank['var'] == var]['p_value'].values[0]
            
            # 배경색 투명도를 0.04로 더 낮춰서 격자선이 돋보이게 함
            graph_bg = "rgba(255, 75, 75, 0.04)" if p_val <= 0.05 else PANEL
            
            with row[i]:
                st.markdown(f"<p style='text-align:left; font-size:12px; font-weight:800; color:{TEXT}; margin-bottom:5px; padding-left:5px;'>{kor_name}</p>", unsafe_allow_html=True)
                
                fig_v = px.violin(df, y=var, color="label", box=True, points=False)
                fig_v.update_layout(
                    height=180, 
                    margin=dict(t=10, b=10, l=35, r=10),
                    showlegend=False, 
                    paper_bgcolor=graph_bg, 
                    plot_bgcolor=graph_bg,
                    font=dict(size=10)
                )
                fig_v.update_xaxes(visible=False)
                # gridcolor를 GRID_VISIBLE로 변경하고 너비를 살짝 올림
                fig_v.update_yaxes(
                    title=None, 
                    showticklabels=True, 
                    gridcolor=GRID_VISIBLE, 
                    gridwidth=1,
                    tickfont=dict(size=9)
                )
                st.plotly_chart(fig_v, use_container_width=True, config={"displayModeBar": False})

        # 2. 리스크 바 차트 영역
        for i, var in enumerate(group):
            kor_name = get_kor_label(var)
            p_val = t_rank[t_rank['var'] == var]['p_value'].values[0]
            
            graph_bg = "rgba(255, 75, 75, 0.04)" if p_val <= 0.05 else PANEL
            
            with row[i+4]:
                st.markdown(f"<p style='text-align:left; font-size:12px; font-weight:800; color:{TEXT}; margin-bottom:5px; padding-left:5px;'>{kor_name} </p>", unsafe_allow_html=True)
                
                df['bin'] = pd.cut(df[var], bins=10)
                risk = df.groupby('bin', observed=False)['label'].mean().reset_index()
                fig_r = px.bar(risk, x=risk.index, y='label', color='label')
                fig_r.update_layout(
                    height=180, 
                    margin=dict(t=10, b=10, l=35, r=10),
                    coloraxis_showscale=False, 
                    paper_bgcolor=graph_bg, 
                    plot_bgcolor=graph_bg,
                    font=dict(size=10)
                )
                fig_r.update_xaxes(visible=False)
                # gridcolor를 GRID_VISIBLE로 변경하고 너비를 살짝 올림
                fig_r.update_yaxes(
                    tickformat=".0%", 
                    title=None, 
                    showticklabels=True, 
                    gridcolor=GRID_VISIBLE, 
                    gridwidth=1,
                    tickfont=dict(size=9)
                )
                st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})