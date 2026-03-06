import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# Defect Post-Analysis Dashboard (Fixes per request)
# =========================================================

st.set_page_config(page_title="Defect Post Analysis", layout="wide")

DATA_PATH = "mice_final_data_with_id.csv"

# ----------------------------
# Theme / spacing
# ----------------------------
SAMSUNG_BLUE = "#1E40FF"
BG = "#F6F8FC"
PANEL = "#FFFFFF"
STROKE = "#E6EAF2"
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

      /* 좌우 여백 최소화 */
      .block-container {{
        padding-top: 0.5rem;
        padding-bottom: 0.9rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        max-width: 100%;
      }}

      /* 섹션 간격 줄이기 */
      hr {{
        border: none;
        border-top: 1px solid var(--stroke);
        margin: 6px 0;
      }}

      /* 카드 공통 */
      .card {{
        background: var(--panel);
        border: 1px solid var(--stroke);
        border-radius: 14px;
        padding: 10px 12px;
        box-shadow: 0 6px 18px rgba(17,24,39,0.06);
      }}

      /* KPI 카드 */
      .kpi {{
        background: var(--panel);
        border: 1px solid var(--stroke);
        border-radius: 14px;
        padding: 12px 14px;
        box-shadow: 0 6px 18px rgba(17,24,39,0.06);
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }}
      .kpi .label {{ color: var(--muted); font-size: 12px; font-weight: 800; }}
      .kpi .value {{ font-size: 40px; font-weight: 950; color: var(--blue); line-height: 1.05; margin-top: 6px; }}
      .kpi .sub {{ color: var(--muted); font-size: 12px; font-weight: 700; margin-top: 10px; }}

      .tiny {{ color: var(--muted); font-size: 12px; font-weight: 700; }}

      /* Plotly modebar(흰 막대) 완전 제거 */
      .modebar, .modebar-container {{ display: none !important; }}
      .js-plotly-plot .plotly .modebar {{ display: none !important; }}
      [data-testid="stPlotlyChart"] .modebar {{ display: none !important; }}

      /* Streamlit 요소 사이 기본 여백 축소 */
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
    fig.update_xaxes(gridcolor="rgba(17,24,39,0.08)", zerolinecolor="rgba(17,24,39,0.10)")
    fig.update_yaxes(gridcolor="rgba(17,24,39,0.08)", zerolinecolor="rgba(17,24,39,0.10)")
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

# id 기준 확실히 정렬
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

STAGE_FEATURES = [c for c in FEATURE_COLS if c.startswith("stage")]

# ----------------------------
# Sidebar (요청한 필터만)
# ----------------------------
st.sidebar.markdown("## 필터")

# 대표 센서 추이용: stage + metric(8개)
stage_pick = st.sidebar.selectbox("Stage", options=STAGES, index=0)
metric_pick = st.sidebar.selectbox("센서 지표(8개)", options=METRICS[:8], index=0)

feature_pick = f"stage{stage_pick}_{metric_pick}"

# 라벨 추이 범위 지정(밀도 해결용)
n_rows = len(df)
default_end = min(n_rows - 1, 8000) if n_rows > 0 else 0
idx_start, idx_end = st.sidebar.slider(
    "라벨 0/1 범위(Index 기준)",
    0,
    max(0, n_rows - 1),
    (0, default_end),
    step=1,
)

# ----------------------------
# Header
# ----------------------------
hL, hR = st.columns([1.3, 2.2], gap="medium")
with hL:
    st.markdown(
        """
        <div style="font-size:24px;font-weight:950;letter-spacing:-0.2px;">불량제품 사후분석</div>
        <div class="tiny">요약 → 영향 변수 → Stage 분포 → 추이</div>
        """,
        unsafe_allow_html=True,
    )
with hR:
    st.markdown(
        f"""
        <div class="card">
          <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;">
            <div>
              <div class="tiny">대표 센서</div>
              <div style="font-weight:900;font-size:14px;">{feature_pick}</div>
            </div>
            <div>
              <div class="tiny">라벨 범위</div>
              <div style="font-weight:900;font-size:14px;">index {idx_start:,} ~ {idx_end:,} (총 {idx_end-idx_start+1:,}개)</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------
# Summary
# ----------------------------
total_prod = int(len(df))
defect_count = int((df["label"] == 1).sum())
ok_count = total_prod - defect_count
dr_all = (defect_count / total_prod * 100) if total_prod else 0.0

df_range = df.iloc[idx_start: idx_end + 1].copy()
range_total = int(len(df_range))
range_defect = int((df_range["label"] == 1).sum())
range_ok = range_total - range_defect

# ----------------------------
# Stats for drivers: Cohen's d + Mann–Whitney p-value(근사)
# ----------------------------
def cohens_d(x_ok: np.ndarray, x_ng: np.ndarray) -> float:
    x_ok = x_ok[np.isfinite(x_ok)]
    x_ng = x_ng[np.isfinite(x_ng)]
    if len(x_ok) < 2 or len(x_ng) < 2:
        return np.nan
    m1, m2 = x_ok.mean(), x_ng.mean()
    s1, s2 = x_ok.std(ddof=1), x_ng.std(ddof=1)
    sp = np.sqrt(((len(x_ok) - 1) * s1 * s1 + (len(x_ng) - 1) * s2 * s2) / (len(x_ok) + len(x_ng) - 2))
    if not np.isfinite(sp) or sp <= 0:
        return np.nan
    return (m2 - m1) / sp

def mannwhitney_pvalue_approx(x_ok: np.ndarray, x_ng: np.ndarray) -> float:
    """
    SciPy 없이 rank 기반 U를 구한 뒤, 정규근사로 two-sided p-value 계산(근사).
    """
    x_ok = x_ok[np.isfinite(x_ok)]
    x_ng = x_ng[np.isfinite(x_ng)]
    n1, n2 = len(x_ok), len(x_ng)
    if n1 < 2 or n2 < 2:
        return np.nan

    x = np.concatenate([x_ok, x_ng])
    ranks = pd.Series(x).rank(method="average").to_numpy()
    r1 = ranks[:n1].sum()

    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sigma <= 0 or not np.isfinite(sigma):
        return np.nan

    z = (u1 - mu) / sigma
    # Phi(|z|) using erf
    import math
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return float(max(min(p, 1.0), 0.0))

# drivers (전체 기준으로 계산)
ok_df = df[df["label"] == 0]
ng_df = df[df["label"] == 1]

driver_rows = []
for c in STAGE_FEATURES:
    x_ok = ok_df[c].to_numpy(dtype=float)
    x_ng = ng_df[c].to_numpy(dtype=float)
    d = cohens_d(x_ok, x_ng)
    p = mannwhitney_pvalue_approx(x_ok, x_ng)
    if np.isfinite(d):
        driver_rows.append((c, d, abs(d), p))

drivers = pd.DataFrame(driver_rows, columns=["feature", "d", "abs_d", "p_value"])
drivers = drivers.sort_values("abs_d", ascending=False).reset_index(drop=True)
top5 = drivers.head(5).copy()

# ----------------------------
# Row A: KPI + KPI + Donut + Drivers (높이 맞춤)
# ----------------------------
a1, a2, a3, a4 = st.columns([1.0, 1.0, 1.2, 1.8], gap="medium")

with a1:
    st.markdown(
        f"""
        <div class="kpi">
          <div class="label">총 생산수 (전체)</div>
          <div class="value">{total_prod:,}</div>
          <div class="sub">정상 {ok_count:,} · 불량 {defect_count:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with a2:
    st.markdown(
        f"""
        <div class="kpi">
          <div class="label">불량률 (전체)</div>
          <div class="value">{dr_all:.2f}%</div>
          <div class="sub">전체 불량수 {defect_count:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with a3:
    st.markdown('<div class="card-title">선택 범위 불량 비율</div>', unsafe_allow_html=True)

    pie_df = pd.DataFrame({"status": ["정상(0)", "불량(1)"], "count": [range_ok, range_defect]})
    fig_pie = px.pie(pie_df, values="count", names="status", hole=0.62)
    fig_pie.update_traces(textposition="inside", textinfo="percent")
    fig_pie.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5))
    fig_pie = style_fig(fig_pie, height=220)
    st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CONFIG)
    st.markdown("</div>", unsafe_allow_html=True)

with a4:
    st.markdown('<div class="card-title">불량 영향 변수 Top 5 (|Cohen’s d| + p-value)</div>', unsafe_allow_html=True)

    if len(top5) == 0:
        st.write("계산 불가(정상/불량 샘플 부족 또는 결측)")
    else:
        plot_df = top5.sort_values("abs_d", ascending=True)
        fig_drv = go.Figure()
        fig_drv.add_trace(
            go.Bar(
                x=plot_df["d"],
                y=plot_df["feature"],
                orientation="h",
                customdata=np.stack([plot_df["abs_d"], plot_df["p_value"]], axis=1),
                hovertemplate="feature=%{y}<br>d=%{x:.3f}<br>|d|=%{customdata[0]:.3f}<br>p≈%{customdata[1]:.3g}<extra></extra>",
            )
        )
        fig_drv.update_layout(xaxis=dict(title="Cohen's d (NG - OK)"), yaxis=dict(title=""))
        fig_drv = style_fig(fig_drv, height=220)
        st.plotly_chart(fig_drv, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------
# Row B: Left(Stage box grouped) / Right(trends)
# ----------------------------
left, right = st.columns([2.1, 1.0], gap="medium")

# ---- Left: Stage boxplot (grouped, color by label)
with left:
    st.markdown('<div class="card-title">Stage별 정상/불량 분포 (8개 지표 · 4×2 facet)</div>', unsafe_allow_html=True)

    # 샘플링(고정, 사이드바로 안 뺌)
    box_sample_n = min(9000, len(df))
    if box_sample_n < len(df):
        idx = np.linspace(0, len(df) - 1, box_sample_n).astype(int)
        df_box = df.iloc[idx].copy()
    else:
        df_box = df.copy()

    rows = []
    for m in METRICS[:8]:
        cols = [f"stage{s}_{m}" for s in STAGES if f"stage{s}_{m}" in df_box.columns]
        if not cols:
            continue
        tmp = df_box[["label"] + cols].melt(id_vars=["label"], var_name="stage_metric", value_name="value")
        tmp["stage"] = tmp["stage_metric"].str.extract(r"stage(\d+)_")[0].astype(int)
        tmp["metric"] = m
        tmp = tmp.dropna(subset=["value"])
        rows.append(tmp[["label", "stage", "metric", "value"]])

    if not rows:
        st.write("박스플롯 생성 데이터가 부족합니다.")
    else:
        long_df = pd.concat(rows, ignore_index=True)
        long_df["label_name"] = long_df["label"].map({0: "정상(0)", 1: "불량(1)"})
        long_df["metric"] = pd.Categorical(long_df["metric"], categories=METRICS[:8], ordered=True)

        fig_box = px.box(
            long_df,
            x="stage",
            y="value",
            color="label_name",
            facet_col="metric",
            facet_col_wrap=4,
            points=False,
            category_orders={"stage": [int(s) for s in STAGES]},
        )
        # facet 제목 정리
        fig_box.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        # 겹치지 않게 stage별 그룹으로(나란히)
        fig_box.update_layout(boxmode="group")
        fig_box.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.10, xanchor="center", x=0.5))
        fig_box.update_xaxes(title_text="stage")
        fig_box.update_yaxes(title_text="value")

        fig_box = style_fig(fig_box, height=560)
        st.plotly_chart(fig_box, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("</div>", unsafe_allow_html=True)

# ---- Right: Representative sensor + Label barcode
with right:
    # Representative sensor trend: ensure id ordering, avoid misleading line jumps
    st.markdown('<div class="card-title">대표 센서 추이 (id 정렬 · 정상/불량 비교)</div>', unsafe_allow_html=True)

    if feature_pick not in df.columns:
        st.write(f"선택 센서 컬럼이 없습니다: {feature_pick}")
    else:
        # 전체를 id 기준 정렬된 df에서 샘플링
        max_points = min(3200, len(df))
        if max_points < len(df):
            sidx = np.linspace(0, len(df) - 1, max_points).astype(int)
            dline = df.iloc[sidx].copy()
        else:
            dline = df.copy()

        dline = dline.dropna(subset=[feature_pick]).copy()

        use_numeric_id = dline["_id_num"].notna().mean() > 0.8
        if use_numeric_id:
            dline = dline.sort_values("_id_num")
            xcol = "_id_num"
            xtitle = "id"
        else:
            dline = dline.reset_index(drop=True)
            dline["_x"] = dline.index
            xcol = "_x"
            xtitle = "index(sorted)"

        ok = dline[dline["label"] == 0]
        ng = dline[dline["label"] == 1]

        fig = go.Figure()
        if len(ok):
            fig.add_trace(go.Scatter(
                x=ok[xcol], y=ok[feature_pick],
                mode="markers", name="정상(0)",
                opacity=0.45, marker=dict(size=4),
            ))
        if len(ng):
            fig.add_trace(go.Scatter(
                x=ng[xcol], y=ng[feature_pick],
                mode="markers", name="불량(1)",
                opacity=0.90, marker=dict(size=6),
            ))

        fig.update_layout(
            xaxis=dict(title=xtitle),
            yaxis=dict(title="sensor value"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig = style_fig(fig, height=280)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("</div>", unsafe_allow_html=True)

    # Label trend: barcode (defects only) + range filter
    st.markdown('<div class="card-title">라벨(0/1) 추이 (불량만 바코드 표시 · 범위 지정)</div>', unsafe_allow_html=True)

    lab = df_range.copy()
    # x축: numeric id 우선
    if lab["_id_num"].notna().mean() > 0.8:
        lab = lab.sort_values("_id_num").reset_index(drop=True)
        xlab = lab["_id_num"]
        xtitle2 = "id"
    else:
        lab = lab.reset_index(drop=True)
        xlab = lab.index
        xtitle2 = "index(range)"

    # 불량만 표시(가독성 최우선)
    defect = lab[lab["label"] == 1]
    fig_lab = go.Figure()

    if len(defect) == 0:
        # 범위 내 불량이 없으면 빈 그래프
        fig_lab.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="선택 범위 내 불량(1) 없음",
            showarrow=False,
            font=dict(color=MUTED, size=12),
        )
    else:
        x_def = defect["_id_num"] if lab["_id_num"].notna().mean() > 0.8 else defect.index

        # 세로선: 각 불량 위치에 vline
        for xv in x_def:
            fig_lab.add_vline(x=xv, line_width=1, opacity=0.35)

        # y는 고정(1)로 찍고, 세로 stem처럼 보이도록 마커 + 짧은 라인
        fig_lab.add_trace(go.Scatter(
            x=x_def,
            y=np.ones(len(defect)),
            mode="markers",
            marker=dict(size=6),
            showlegend=False,
            hovertemplate="x=%{x}<br>label=1<extra></extra>",
        ))

    fig_lab.update_layout(
        xaxis=dict(title=xtitle2),
        yaxis=dict(title="", range=[0.0, 1.2], tickmode="array", tickvals=[1], ticktext=["1(불량)"]),
        showlegend=False,
    )
    fig_lab = style_fig(fig_lab, height=220)
    st.plotly_chart(fig_lab, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("</div>", unsafe_allow_html=True)