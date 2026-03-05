import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import re

# =========================================================
# Process Monitor (Latest) + Control-Limit Anomaly
# - 사후 분석용 메인 대시보드 (1페이지)
# - 불량 예측 및 SHAP 완전 제거됨
# - Stage Snapshot: 현업 SCADA 스타일 (원형 게이지 + 전체 배경 색상)
# =========================================================

st.set_page_config(page_title="공정 실시간 모니터링", layout="wide")

DATA_PATH = "mice_final_data_with_id.csv"

# 관리도 기반 한계선(기본): mean ± k*sigma
CTRL_K = 3.0
RECENT_WINDOW_N = 120

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

df = load_csv(DATA_PATH)

if "id" not in df.columns or "label" not in df.columns:
    st.error("CSV에 'id', 'label' 컬럼이 필요합니다.")
    st.stop()

# deterministic order then pick last
df = df.copy()
id_num = pd.to_numeric(df["id"], errors="coerce")
df["_id_sort"] = id_num if id_num.notna().mean() > 0.8 else df["id"].astype(str)
df = df.sort_values("_id_sort").reset_index(drop=True)
df["run"] = np.arange(len(df))

FEATURE_COLS = [c for c in df.columns if c not in ("id", "label", "_id_sort", "run")]

# 영어 변수명 -> 한글 매핑 딕셔너리
SENSOR_KO = {
    "temp": "온도", "humidity": "습도", 
    "flow_deviation": "유량 편차", "density_deviation": "밀도 편차", 
    "viscosity_deviation": "점도 편차", "o2_deviation": "O2 편차", 
    "n_deviation": "N2 편차", "co2_deviation": "CO2 편차",
    "run": "생산 순번"
}

# Feature 이름 한글 변환 함수 (전역)
def translate_feature_name(f_name):
    m_diff = re.match(r"^(?P<metric>.+)_diff_(?P<i>\d+)_(?P<j>\d+)$", f_name)
    if m_diff:
        metric = m_diff.group("metric")
        i = m_diff.group("i")
        j = m_diff.group("j")
        metric_ko = SENSOR_KO.get(metric, metric)
        return f"S{i}-S{j} {metric_ko}"
    
    m_slope = re.match(r"^(?P<metric>.+)_slope$", f_name)
    if m_slope:
        metric = m_slope.group("metric")
        metric_ko = SENSOR_KO.get(metric, metric)
        return f"{metric_ko} 기울기"

    if f_name.startswith("stage"):
        parts = f_name.split("_", 1)
        if len(parts) == 2:
            stage_num = parts[0][-1]
            metric = SENSOR_KO.get(parts[1], parts[1])
            return f"공정{stage_num} {metric}"
            
    return SENSOR_KO.get(f_name, f_name)

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

STAGES, METRICS = infer_stages_metrics(FEATURE_COLS)

def fcol(stage: int, metric: str) -> str:
    return f"stage{stage}_{metric}"

def to_num(x):
    if isinstance(x, pd.DataFrame):
        return x.apply(pd.to_numeric, errors="coerce")
    return pd.to_numeric(x, errors="coerce")

if not STAGES:
    st.error("stage1~5 구조를 찾지 못했습니다.")
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
METRICS = [m for m in CORE_METRICS_ORDER if m in METRICS] + [m for m in METRICS if m not in CORE_METRICS_ORDER]
DEV_METRICS = [m for m in METRICS if m.endswith("_deviation")]

# ----------------------------
# Styling (White Theme)
# ----------------------------
SAMSUNG_BLUE_2 = "#1428A0"
GOOD = "#10B981"
WARN = "#F59E0B"
BAD = "#DC2626"
TEXT = "#111827"
MUTED = "#6B7280"

BG = "#F9FAFB"
PANEL = "#FFFFFF"
STROKE = "#E5E7EB"

st.markdown(
    f"""
    <style>
      :root {{
        --bg:{BG}; --panel:{PANEL}; --stroke:{STROKE};
        --text:{TEXT}; --muted:{MUTED};
        --blue2:{SAMSUNG_BLUE_2};
      }}
      .stApp {{ background: var(--bg); color: var(--text); }}
      header, [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
      footer {{ visibility: hidden; }}

      [data-testid="stSidebar"] {{
        background-color: #F8FAFC;
        border-right: 1px solid #E5E7EB;
      }}

      .topbar {{
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      }}
      .brand {{ font-size: 24px; font-weight: 900; letter-spacing: 0.4px; color: var(--blue2); }}
      .subtitle {{ color: var(--muted); font-weight: 700; margin-top: 2px; }}
      .rightmeta {{ text-align: right; font-weight: 900; }}

      .kpi {{
        background: var(--panel);
        border: 1px solid var(--stroke);
        border-radius: 12px;
        padding: 12px 14px;
        height: 84px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
      }}
      .kpi .l {{ color: var(--muted); font-size: 12px; font-weight: 800; }}
      .kpi .v {{ font-size: 30px; font-weight: 900; color: var(--blue2); line-height: 1.05; }}
      .kpi .u {{ color: var(--muted); font-size: 13px; font-weight: 800; margin-left: 6px; }}

      .panel {{
        background: var(--panel);
        border: 1px solid var(--stroke);
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
      }}
      .pt {{ font-weight: 900; margin-bottom: 8px; }}
      .pill {{
        display:inline-block; padding: 2px 10px; border-radius:999px;
        border: 1px solid rgba(20,40,160,0.2);
        background: rgba(20,40,160,0.05);
        color: var(--blue2); font-size: 12px; font-weight: 800; margin-left: 8px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

def kpi(container, label, value, unit="", fmt="{:.2f}"):
    v = "-" if value is None or (isinstance(value, float) and np.isnan(value)) else fmt.format(value)
    container.markdown(
        f'<div class="kpi"><div class="l">{label}</div><div><span class="v">{v}</span><span class="u">{unit}</span></div></div>',
        unsafe_allow_html=True,
    )

def fmtv(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{float(x):.{nd}f}"

# ----------------------------
# Latest sample + recent window
# ----------------------------
t = int(df["run"].iloc[-1])
cur = df.iloc[-1]

w0 = max(0, t - RECENT_WINDOW_N + 1)
hist = df.iloc[w0 : t + 1].copy()

# ----------------------------
# Control limits (관리도 기반): mean ± k*sigma
# ----------------------------
def compute_ctrl_limits(hist_df: pd.DataFrame, feature_cols: list[str], k: float) -> dict[str, tuple[float, float, float, float]]:
    limits = {}
    for c in feature_cols:
        s = to_num(hist_df[c])
        mu = float(np.nanmean(s)) if np.isfinite(np.nanmean(s)) else np.nan
        sd = float(np.nanstd(s, ddof=1)) if np.isfinite(np.nanstd(s, ddof=1)) else np.nan
        if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 0:
            limits[c] = (mu, sd, np.nan, np.nan)
        else:
            limits[c] = (mu, sd, mu - k * sd, mu + k * sd)
    return limits

CTRL_LIMITS = compute_ctrl_limits(hist, FEATURE_COLS, CTRL_K)

def is_over_limit(val, feature: str) -> bool:
    if feature not in CTRL_LIMITS:
        return False
    mu, sd, lcl, ucl = CTRL_LIMITS[feature]
    if not np.isfinite(lcl) or not np.isfinite(ucl):
        return False
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    v = float(val)
    return (v < lcl) or (v > ucl)

def limit_exceed_score(val, feature: str) -> float:
    if feature not in CTRL_LIMITS:
        return 0.0
    mu, sd, lcl, ucl = CTRL_LIMITS[feature]
    if val is None or (isinstance(val, float) and np.isnan(val)) or (not np.isfinite(mu)) or (not np.isfinite(sd)) or sd <= 0:
        return 0.0
    return abs((float(val) - mu) / sd)

def count_equipment_anomalies(df_window: pd.DataFrame) -> int:
    cnt = 0
    for _, r in df_window.iterrows():
        hit = False
        for c in FEATURE_COLS:
            v = pd.to_numeric(r.get(c), errors="coerce")
            if is_over_limit(v, c):
                hit = True
                break
        cnt += int(hit)
    return cnt

equip_anom_recent = count_equipment_anomalies(hist)

# ----------------------------
# KPI 데이터 준비
# ----------------------------
total_prod = int(len(df))
defect_count = int((df["label"] == 1).sum())
defect_rate = (defect_count / total_prod) if total_prod > 0 else np.nan
equip_anom_count = int(equip_anom_recent)

# ----------------------------
# Top bar
# ----------------------------
top_l, top_r = st.columns([1.5, 1.5])
with top_l:
    st.markdown(
        """
        <div class="topbar">
          <div class="brand">SAMSUNG ELECTRONICS · 공정 모니터링</div>
          <div class="subtitle">5단계 공정 품질 제어 · 최신 데이터 뷰</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_r:
    st.markdown(
        f"""
        <div class="topbar">
          <div class="rightmeta" style="font-size:18px;">현재 생산 제품 현황</div>
          <div class="subtitle rightmeta">진행 순번={t} · 제품 ID={cur["id"]} · 분석 구간={w0}~{t} ({len(hist)}건)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# KPI row
# ----------------------------
k1, k2, k3, k4 = st.columns(4)
kpi(k1, "총 생산량", float(total_prod), "개", "{:.0f}")
kpi(k2, "불량 제품 수", float(defect_count), "개", "{:.0f}")
kpi(k3, "불량 비율", defect_rate * 100 if np.isfinite(defect_rate) else np.nan, "%", "{:.2f}")
kpi(k4, "최근 설비 이상 감지", float(equip_anom_count), "건", "{:.0f}")

st.markdown("")

# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([2.55, 1.45], gap="medium")

with left:
    st.markdown('<div class="panel"><div class="pt">공정별 실시간 상태 요약 (SCADA View) <span class="pill">전체 배경 색칠 모드</span></div></div>', unsafe_allow_html=True)

    stage_metric_options = ["[자동] 공정별 가장 위험한 편차"] + [SENSOR_KO.get(m, m) for m in DEV_METRICS]
    selected_stage_metric = st.selectbox("공정 게이지 모니터링 지표 선택", stage_metric_options)

    cols = st.columns(len(STAGES), gap="small")

    def pick_worst_metric_for_stage(stage: int) -> str:
        candidates = []
        for m in DEV_METRICS:
            f = fcol(stage, m)
            if f in df.columns:
                v = pd.to_numeric(cur.get(f), errors="coerce")
                score = limit_exceed_score(v, f)
                over = is_over_limit(v, f)
                candidates.append(((10.0 if over else 0.0) + score, m))
        candidates.sort(reverse=True)
        if candidates: return candidates[0][1]
        return DEV_METRICS[0]

    # 🚀 완벽한 SCADA 스타일 카드 렌더링 루프
    for i, s in enumerate(STAGES):
        if selected_stage_metric == "[자동] 공정별 가장 위험한 편차":
            target_m = pick_worst_metric_for_stage(s)
        else:
            target_m = next(k for k, v in SENSOR_KO.items() if v == selected_stage_metric)

        target_f = fcol(s, target_m)
        val = pd.to_numeric(cur.get(target_f), errors="coerce")
        metric_ko = SENSOR_KO.get(target_m, target_m)
        
        mu, sd, lcl, ucl = CTRL_LIMITS.get(target_f, (0, 0, 0, 0))
        
        # 색상 및 상태 판별 (Z-score 기반)
        bg_color = GOOD
        pct = 0
        if pd.notna(val) and sd > 0:
            z = abs(val - mu) / sd
            if z >= 3:
                bg_color = BAD
            elif z >= 2:
                bg_color = WARN
            # 4시그마를 100%로 잡고 링(Donut)을 얼마나 채울지 퍼센트 계산
            pct = min((z / 4.0) * 100, 100)
            
        val_str = f"{val:.2f}" if pd.notna(val) else "-"

        # 🚀 순수 SVG 미니 바 차트 (최근 10건) 생성
        recent_10 = hist[target_f].tail(10).values
        svg_bars = ""
        if len(recent_10) > 0:
            min_v, max_v = np.nanmin(recent_10), np.nanmax(recent_10)
            n_vals = len(recent_10)
            for j, v in enumerate(recent_10):
                if pd.isna(v): continue
                if max_v == min_v:
                    norm_h = 15
                else:
                    norm_h = max(2, ((v - min_v) / (max_v - min_v)) * 30)
                
                # 우측 정렬되도록 x축 계산
                x = (10 - n_vals + j) * 10 + 1.5
                y = 30 - norm_h
                
                # 3시그마 벗어난 과거 데이터는 하얀색 막대, 정상은 반투명 하얀색으로 깔끔하게
                bar_color = "#ffffff" if is_over_limit(v, target_f) else "rgba(255,255,255,0.4)"
                svg_bars += f'<rect x="{x}" y="{y}" width="7" height="{norm_h}" fill="{bar_color}" rx="1" />'

        # 🚀 들여쓰기(Indentation) 제거: Streamlit Markdown 파서 오류 방지용 완벽한 한 줄 HTML
        html_card = f"""<div style="background-color: {bg_color}; border-radius: 12px; padding: 15px; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 210px; display: flex; flex-direction: column; justify-content: space-between;"><div><div style="font-weight: 900; font-size: 16px;">공정 {s}</div><div style="font-size: 13px; opacity: 0.9; margin-bottom: 5px;">{metric_ko}</div></div><div style="flex-grow: 1; display: flex; align-items: center; justify-content: center;"><svg viewBox="0 0 36 36" style="width: 85px; height: 85px; overflow: visible;"><path style="fill:none; stroke:rgba(255,255,255,0.25); stroke-width:3.5;" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" /><path style="fill:none; stroke:#ffffff; stroke-width:3.5; stroke-linecap:round; stroke-dasharray:{pct}, 100;" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" /><text x="18" y="21.5" style="fill:#ffffff; font-size:8.5px; font-weight:bold; text-anchor:middle;">{val_str}</text></svg></div><div style="height: 30px; width: 100%; margin-top: 5px;"><svg width="100%" height="100%" viewBox="0 0 100 30" preserveAspectRatio="none">{svg_bars}</svg></div></div>"""

        with cols[i]:
            st.markdown(html_card, unsafe_allow_html=True)

    st.markdown("")

    # ----------------------------
    # Recent Trend 
    # ----------------------------
    st.markdown('<div class="panel"><div class="pt">Recent Trend <span class="pill">한계선 초과(3σ) 집중 모니터링</span></div></div>', unsafe_allow_html=True)

    def pick_most_critical_feature() -> str:
        scored = []
        exceeded = []
        for c in FEATURE_COLS:
            v = pd.to_numeric(cur.get(c), errors="coerce")
            s = limit_exceed_score(v, c)
            scored.append((s, c))
            if is_over_limit(v, c):
                exceeded.append((s, c))
        exceeded.sort(reverse=True)
        scored.sort(reverse=True)
        if exceeded:
            return exceeded[0][1]
        return scored[0][1] if scored else "run"

    auto_critical_feat = pick_most_critical_feature()
    trend_options_dict = {f: translate_feature_name(f) for f in FEATURE_COLS if "stage" in f}
    
    default_stage = 1
    default_metric = METRICS[0] if METRICS else "temp"
    
    if auto_critical_feat and auto_critical_feat.startswith("stage"):
        _parts = auto_critical_feat.split("_", 1)
        if len(_parts) == 2 and _parts[0][5:].isdigit():
            default_stage = int(_parts[0][5:])
            default_metric = _parts[1]

    s_idx = STAGES.index(default_stage) if default_stage in STAGES else 0
    m_idx = METRICS.index(default_metric) if default_metric in METRICS else 0

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_stage = st.selectbox("분석 공정 (Stage) 선택", STAGES, index=s_idx)
    with filter_col2:
        selected_metric = st.selectbox("분석 지표 선택", METRICS, format_func=lambda x: SENSOR_KO.get(x, x), index=m_idx)
    
    trend_feature = fcol(selected_stage, selected_metric)

    plot = hist[["run", "label", "id"]].copy()
    plot[trend_feature] = to_num(hist[trend_feature]) if trend_feature in hist.columns else np.nan
    plot = plot.dropna(subset=[trend_feature])

    mu, sd, lcl, ucl = CTRL_LIMITS.get(trend_feature, (np.nan, np.nan, np.nan, np.nan))

    fig = go.Figure()
    translated_trend_feature = translate_feature_name(trend_feature)

    fig.add_trace(go.Scatter(x=plot["run"], y=plot[trend_feature], mode="lines+markers", name=translated_trend_feature, line=dict(color=SAMSUNG_BLUE_2)))

    if np.isfinite(lcl) and np.isfinite(ucl):
        out_of_control = plot[(plot[trend_feature] > ucl) | (plot[trend_feature] < lcl)]
        if not out_of_control.empty:
            fig.add_trace(
                go.Scatter(
                    x=out_of_control["run"],
                    y=out_of_control[trend_feature],
                    mode="markers",
                    name="한계선(3σ) 이탈",
                    marker=dict(size=10, color=BAD, symbol="circle-open", line=dict(width=2)),
                )
            )

    if t in set(plot["run"]):
        ycur = float(plot.loc[plot["run"] == t, trend_feature].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=[t],
                y=[ycur],
                mode="markers+text",
                name="현재 위치",
                text=["현재 제품"],
                textposition="top center",
                marker=dict(size=12, symbol="diamond", color=WARN),
            )
        )

    if np.isfinite(lcl) and np.isfinite(ucl):
        fig.add_hline(y=ucl, line_dash="dot", line_color=BAD, annotation_text=f"UCL (+3σ): {ucl:.2f}", annotation_font_color=BAD)
        fig.add_hline(y=lcl, line_dash="dot", line_color=BAD, annotation_text=f"LCL (-3σ): {lcl:.2f}", annotation_font_color=BAD)

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        xaxis=dict(title="생산 순번", gridcolor="rgba(0,0,0,0.05)"),
        yaxis=dict(title=translated_trend_feature, gridcolor="rgba(0,0,0,0.05)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    # ----------------------------
    # Right (1): 8 mini charts 
    # ----------------------------
    st.markdown('<div class="panel"><div class="pt">Stage Profile <span class="pill">8개 지표 현황</span></div></div>', unsafe_allow_html=True)

    MINI_METRICS = [
        "temp",
        "humidity",
        "flow_deviation",
        "density_deviation",
        "viscosity_deviation",
        "o2_deviation",
        "n_deviation",
        "co2_deviation",
    ]
    MINI_METRICS = [m for m in MINI_METRICS if m in METRICS] 

    def stage_series_for_metric(metric: str):
        xs = []
        ys = []
        for s in STAGES:
            f = fcol(s, metric)
            if f in df.columns:
                xs.append(s)
                ys.append(pd.to_numeric(cur.get(f), errors="coerce"))
            else:
                xs.append(s)
                ys.append(np.nan)
        return xs, ys

    grid_cols = st.columns(4, gap="small")
    figs = []

    for m in MINI_METRICS[:8]:
        xs, ys = stage_series_for_metric(m)
        metric_ko = SENSOR_KO.get(m, m)

        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=metric_ko, line=dict(color=SAMSUNG_BLUE_2)))
        fig_m.update_layout(
            height=170,
            margin=dict(l=6, r=6, t=25, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, size=11),
            xaxis=dict(title="공정", tickmode="array", tickvals=xs, gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
            showlegend=False,
            title=dict(text=metric_ko, x=0.02, y=0.98, xanchor="left", yanchor="top"),
        )
        figs.append(fig_m)

    for idx, fig_m in enumerate(figs):
        col = grid_cols[idx % 4]
        with col:
            st.plotly_chart(fig_m, use_container_width=True)
