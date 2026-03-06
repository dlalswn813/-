# app.py
# 단일 화면(사이드바 제거) 고정 대시보드 버전
# - 페이지 스크롤: 차단(overflow hidden)
# - 모든 섹션: 고정 높이로 배치(Plotly height 지정)
# - 알람/표는 "페이지 스크롤"이 아니라 "컴포넌트 내부"에서만 스크롤(높이 고정)
#
# 실행:
#   streamlit run app.py
#
# 같은 폴더에 아래 파일이 있어야 함:
#   catboost_final_model.cbm
#   mice_final_data_with_id.csv

import os
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from catboost import CatBoostClassifier, Pool

# 선택 가능한 알람 테이블(클릭)용
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# =========================
# Streamlit page config
# =========================
st.set_page_config(
    page_title="반도체 세정 공정 모니터링",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================
# CSS: 사이드바 숨김 + 전체 페이지 스크롤 차단 + 간격 축소
# =========================
st.markdown(
    """
<style>
/* sidebar 완전 제거 */
[data-testid="stSidebar"] { display:none !important; }
header { visibility: hidden; height: 0px; }
#MainMenu { visibility: hidden; }

/* 전체 스크롤 차단 (한 화면 고정) */
html, body { height: 100%; overflow: hidden; }
section.main { height: 100vh; overflow: hidden; }
div.block-container { height: 100vh; overflow: hidden; padding-top: 10px; padding-bottom: 6px; }

/* 섹션 간격 압축 */
[data-testid="stVerticalBlock"] { gap: 8px; }
.stDivider { margin: 6px 0 !important; }

/* 제목 */
.main-title{
  font-size: 20px;
  font-weight: 900;
  color: #1f2d3d;
  margin: 0;
  padding: 0;
}
.subhint{
  font-size: 12px;
  color: rgba(49,51,63,0.65);
  margin-top: -2px;
}

/* KPI 카드 */
.kpi-card{
  background:#f8f9fa;
  border:1px solid #e6e6e6;
  border-radius:12px;
  padding:10px 12px;
  height: 86px; /* 화면 고정 */
  display:flex;
  flex-direction:column;
  justify-content:space-between;
}
.kpi-label{ font-size:12px; color:#666; font-weight:700; }
.kpi-value{ font-size:22px; color:#1f2d3d; font-weight:900; line-height: 1.0; }
.kpi-foot{ font-size:12px; font-weight:800; }
.kpi-up{ color:#ff4b4b; }
.kpi-down{ color:#00cc66; }
.kpi-zero{ color:#999; }

/* 패널 제목 */
.panel-title{
  font-size: 14px;
  font-weight: 900;
  margin: 0 0 4px 0;
  color: #222;
}

/* 상단 필터 row 한 줄 고정 느낌 */
.filter-wrap [data-testid="stHorizontalBlock"] { gap: 10px; }
.filter-wrap label { font-size: 12px !important; }
.filter-wrap [data-testid="stDateInput"] > div { min-width: 140px; }
.filter-wrap [data-testid="stSelectbox"] > div { min-width: 140px; }

/* AgGrid 높이 고정 */
.ag-theme-streamlit { border-radius: 10px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Paths
# =========================
HERE = Path(__file__).resolve()
ROOT_DIR = HERE.parent
MODEL_PATH = str(ROOT_DIR / "catboost_final_model.cbm")
DATA_PATH = str(ROOT_DIR / "mice_final_data_with_id.csv")

# =========================
# Constants: 화면 고정 높이(px)
# =========================
H_LINE = 220          # "실시간 불량 예측 확률" 라인
H_GAUGE = 220         # 게이지
H_TABLE = 175         # 공정 변수 테이블
H_TREND = 190         # 공정 변수 트렌드
H_HEAT = 190          # anomaly score heatmap
H_BAR = 190           # 제품 불량 영향도 bar
H_ALARM = 460         # 우측 알람 테이블(전체 mid+bottom 합)

# =========================
# Feature mapping / utils
# =========================
_STAGE_RE = re.compile(r"^stage(\d+)_(.+)$", re.IGNORECASE)

SENSOR_SUFFIX_TO_KO = {
    "temp": "온도",
    "humidity": "습도",
    "flow_deviation": "유량 편차",
    "density_deviation": "밀도 편차",
    "viscosity_deviation": "점도 편차",
    "co2_deviation": "CO₂ 편차",
    "o2_deviation": "O₂ 편차",
    "n_deviation": "N₂ 편차",
}
KO_TO_SUFFIX = {v: k for k, v in SENSOR_SUFFIX_TO_KO.items()}

def parse_stage(sensor_col: str):
    m = _STAGE_RE.match(sensor_col)
    return int(m.group(1)) if m else None

def parse_suffix(sensor_col: str):
    m = _STAGE_RE.match(sensor_col)
    return m.group(2) if m else None

def parse_process(sensor_col: str) -> str:
    n = parse_stage(sensor_col)
    return f"Stage {n}" if n is not None else "-"

def sensor_display_name(sensor_col: str) -> str:
    suf = parse_suffix(sensor_col)
    if suf and suf in SENSOR_SUFFIX_TO_KO:
        return SENSOR_SUFFIX_TO_KO[suf]
    return sensor_col

def is_id_col(c: str) -> bool:
    c2 = c.lower()
    return c2 == "id" or c2.endswith("_id") or c2 == "product_id"

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def get_delta_class(diff: float):
    if diff > 0.01:
        return "kpi-up", f"▲ {abs(diff):.2f}"
    if diff < -0.01:
        return "kpi-down", f"▼ {abs(diff):.2f}"
    return "kpi-zero", "─ 0.00"

def translate_feature(col: str) -> str:
    manual_map = {
        "viscosity_deviation_diff_3_2": "S3-S2 점도편차 차이",
        "flow_deviation_diff_5_4": "S5-S4 유량편차 차이",
        "viscosity_deviation_diff_5_1": "S5-S1 점도편차 차이",
        "co2_deviation_diff_5_1": "S5-S1 CO2편차 차이",
        "n_deviation_slope": "N2 편차 기울기",
        "temp_deviation": "전체 온도 편차",
        "hum_deviation": "전체 습도 편차",
    }
    if col in manual_map:
        return manual_map[col]

    base_sensor_dict = {
        "temp": "온도",
        "humidity": "습도",
        "flow": "유량",
        "density": "밀도",
        "viscosity": "점도",
        "co2": "CO2",
        "o2": "O2",
        "n": "N2",
    }

    if col.startswith("stage"):
        parts = col.split("_")
        stage_num = parts[0].replace("stage", "S")
        sensor_eng = parts[1] if len(parts) > 1 else ""
        sensor_kor = base_sensor_dict.get(sensor_eng, sensor_eng)
        is_dev = " 편차" if "deviation" in col else ""
        return f"{stage_num} {sensor_kor}{is_dev}"

    return col.replace("_", " ")

def get_h_score(df: pd.DataFrame) -> float:
    numeric_cols = df.select_dtypes(include=["number"]).columns
    numeric_cols = [c for c in numeric_cols if c != "label"]

    normal_df = df[df["label"] == 0][numeric_cols] if "label" in df.columns else df[numeric_cols]
    if normal_df.empty:
        normal_df = df[numeric_cols]

    stats = normal_df.agg(["mean", "std"]).T
    recent_df = df[numeric_cols].tail(30)

    total_anomalies = 0
    for col in numeric_cols:
        mu = stats.loc[col, "mean"]
        sigma = stats.loc[col, "std"]
        ucl = mu + (3 * sigma)
        lcl = mu - (3 * sigma)
        anomalies = recent_df[col][(recent_df[col] > ucl) | (recent_df[col] < lcl)].count()
        total_anomalies += anomalies

    return total_anomalies / len(numeric_cols) if len(numeric_cols) else 0.0

def calc_imr_limits(values: np.ndarray):
    x = values[np.isfinite(values)]
    if len(x) < 3:
        return dict(mean=np.nan, ucl=np.nan, lcl=np.nan, mrbar=np.nan, sigma=np.nan)
    mean = float(np.mean(x))
    mr = np.abs(np.diff(x))
    mrbar = float(np.mean(mr)) if len(mr) else np.nan
    d2 = 1.128
    sigma = float(mrbar / d2) if np.isfinite(mrbar) else np.nan
    return dict(mean=mean, ucl=mean + 3 * sigma, lcl=mean - 3 * sigma, mrbar=mrbar, sigma=sigma)

@st.cache_data(show_spinner=False)
def build_limits(df_in: pd.DataFrame, sensors: list[str]) -> dict:
    out = {}
    for s in sensors:
        v = df_in[s].to_numpy(dtype=float)
        out[s] = calc_imr_limits(v)
    return out

def build_alarm_table(df_in: pd.DataFrame, id_col: str, sensors: list[str], limits_map: dict, max_rows: int = 2000):
    tmp = df_in.copy()
    tmp = tmp.sort_values(id_col).reset_index(drop=True)
    tmp["order"] = np.arange(len(tmp))
    tmp = tmp.rename(columns={id_col: "id"})

    rows = []
    for s in sensors:
        lim = limits_map.get(s, {})
        ucl, lcl = lim.get("ucl", np.nan), lim.get("lcl", np.nan)
        if not (np.isfinite(ucl) and np.isfinite(lcl)):
            continue

        v = tmp[s].to_numpy(dtype=float)
        ooc = np.isfinite(v) & ((v > ucl) | (v < lcl))
        idxs = np.where(ooc)[0]

        for i in idxs:
            if len(rows) >= max_rows:
                break
            if np.isfinite(v[i]) and v[i] > ucl:
                direction = "+"
                dist = float(abs(v[i] - ucl))
            else:
                direction = "-"
                dist = float(abs(v[i] - lcl))

            rows.append(
                {
                    "event_id": f"{s}::{tmp.loc[i,'id']}",
                    "공정 단계": parse_process(s),
                    "측정 항목": sensor_display_name(s),
                    "sensor_raw": s,
                    "제품 ID": str(tmp.loc[i, "id"]),
                    "order": int(tmp.loc[i, "order"]),
                    "이탈폭": f"{direction}{dist:.3f}",
                    "측정값": float(v[i]) if np.isfinite(v[i]) else np.nan,
                }
            )

    alarms = pd.DataFrame(rows)
    if alarms.empty:
        return alarms
    alarms = alarms.sort_values(["order"], ascending=[False]).reset_index(drop=True)
    return alarms

# =========================
# Load model + data
# =========================
@st.cache_resource(ttl=10)
def load_all():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        return None, None, None, None, None, None, "파일을 찾을 수 없습니다."

    try:
        model = CatBoostClassifier().load_model(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)

        # id 컬럼 탐색
        cols = df.columns.tolist()
        id_col = next((c for c in cols if is_id_col(c)), None)
        if id_col is None:
            return None, None, None, None, None, None, "ID 컬럼을 찾을 수 없습니다."

        # 숫자 안정화(필요한 범위만)
        for c in df.columns:
            if c != id_col:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # 모델 피처
        features = model.feature_names_

        # 파생변수 5개
        df["viscosity_deviation_diff_3_2"] = df.get("stage3_viscosity_deviation", 0) - df.get("stage2_viscosity_deviation", 0)
        df["flow_deviation_diff_5_4"] = df.get("stage5_flow_deviation", 0) - df.get("stage4_flow_deviation", 0)
        df["viscosity_deviation_diff_5_1"] = df.get("stage5_viscosity_deviation", 0) - df.get("stage1_viscosity_deviation", 0)
        df["co2_deviation_diff_5_1"] = df.get("stage5_co2_deviation", 0) - df.get("stage1_co2_deviation", 0)

        n_cols = [f"stage{i}_n_deviation" for i in range(1, 6)]
        if all(c in df.columns for c in n_cols):
            df["n_deviation_slope"] = np.polyfit(range(5), df[n_cols].values.T, 1)[0]
        else:
            df["n_deviation_slope"] = 0

        # 누락 피처 0 채움
        for col in features:
            if col not in df.columns:
                df[col] = 0

        X = df[features].fillna(0)

        df["prob"] = (model.predict_proba(X)[:, 1] * 100).round(1)

        # 최신 SHAP
        latest_pool = Pool(X.tail(1))
        latest_shap = model.get_feature_importance(data=latest_pool, type="ShapValues")[0, :-1]

        # h_score
        h_score = get_h_score(df.fillna(0))

        return df, X, model, features, latest_shap, id_col, None
    except Exception as e:
        return None, None, None, None, None, None, str(e)

df, X, model, feature_names, shap_vals_latest, id_col, error_msg = load_all()
if error_msg:
    st.error(error_msg)
    st.stop()

# =========================
# Filters (상단)
# =========================
# 날짜/설비/품번은 "예시" 형태로 UI만 잡고,
# 실제 데이터 컬럼에 맞게 필터링은 너가 컬럼명만 연결하면 됨.
st.markdown('<div class="main-title">반도체 세정 공정 실시간 품질 모니터링</div>', unsafe_allow_html=True)
st.markdown('<div class="subhint">단일 화면 고정 대시보드(사이드바 제거)</div>', unsafe_allow_html=True)

st.markdown('<div class="filter-wrap">', unsafe_allow_html=True)
f1, f2, f3, f4, f5 = st.columns([1.7, 1.4, 1.4, 2.2, 1.0], gap="small")

with f1:
    date_range = st.date_input(
        "기간",
        value=(datetime.today().date(), datetime.today().date()),
        label_visibility="visible",
    )

with f2:
    equip = st.selectbox("설비", options=["전체", "F01", "F02", "F03"], index=2)

with f3:
    part = st.selectbox("품번", options=["전체", "992-512", "958-512", "936-K12", "921-23"], index=0)

with f4:
    st.caption(" ")
    st.write("")  # spacer

with f5:
    st.caption(" ")
    st.button("화면", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# KPI row
# =========================
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

temp_cols = [f"stage{i}_temp" for i in range(1, 6) if f"stage{i}_temp" in df.columns]
hum_cols = [f"stage{i}_humidity" for i in range(1, 6) if f"stage{i}_humidity" in df.columns]

curr_fail_rate = float(df["label"].mean() * 100) if "label" in df.columns else 0.0
prev_fail_rate = float(df.iloc[:-1]["label"].mean() * 100) if ("label" in df.columns and len(df) > 1) else curr_fail_rate

diff_fail = curr_fail_rate - prev_fail_rate
diff_temp = float(latest[temp_cols].mean() - prev[temp_cols].mean()) if temp_cols else 0.0
diff_hum = float(latest[hum_cols].mean() - prev[hum_cols].mean()) if hum_cols else 0.0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5, gap="small")

# 생산수량은 label/데이터 구조에 따라 다를 수 있어 기본은 행수
prod_cnt = len(df)
ng_cnt = int(df["label"].sum()) if "label" in df.columns else 0
fail_prob = float(latest.get("prob", 0.0))

cls_fail, txt_fail = get_delta_class(diff_fail)
cls_temp, txt_temp = get_delta_class(diff_temp)
cls_hum, txt_hum = get_delta_class(diff_hum)

with kpi1:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">생산 수량</div>
  <div class="kpi-value">{prod_cnt:,} 개</div>
  <div class="kpi-foot kpi-zero"> </div>
</div>
""",
        unsafe_allow_html=True,
    )

with kpi2:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">NG 수량(불량 건수)</div>
  <div class="kpi-value">{ng_cnt:,} 개</div>
  <div class="kpi-foot kpi-zero"> </div>
</div>
""",
        unsafe_allow_html=True,
    )

with kpi3:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">NG 비율</div>
  <div class="kpi-value">{curr_fail_rate:.2f}%</div>
  <div class="kpi-foot {cls_fail}">{txt_fail}</div>
</div>
""",
        unsafe_allow_html=True,
    )

h_score = get_h_score(df.fillna(0))

with kpi4:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">설비 이상 감지(개)</div>
  <div class="kpi-value">{h_score:.0f}</div>
  <div class="kpi-foot kpi-zero"> </div>
</div>
""",
        unsafe_allow_html=True,
    )

with kpi5:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">불량 예측 확률(최신)</div>
  <div class="kpi-value">{fail_prob:.1f}%</div>
  <div class="kpi-foot kpi-zero"> </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.divider()

# =========================
# 센서/공정 변수 요약 테이블 (예시: temp/humidity/flow...)
# =========================
def build_process_summary(df_in: pd.DataFrame):
    # "공정 변수"로 보여줄 컬럼을 여기서 선택(네 데이터에 맞게 수정 가능)
    # 예시: cycle_time 같은 게 없을 수도 있어서, 존재하는 것만 넣음.
    candidates = [
        "cycle_time",
        "injection_time",
        "cushion_min",
        "fill_speed_p",
        "range_temp1",
    ]
    # stage 평균으로 보여줄 수도 있으니, 데이터에 맞춰 선택
    # 여기서는 우선 df에 존재하는 것만 사용
    cols = [c for c in candidates if c in df_in.columns]
    if not cols:
        # 최소 안전: 숫자형 상위 일부
        num_cols = df_in.select_dtypes(include=["number"]).columns.tolist()
        cols = [c for c in num_cols if c not in ["label", "prob"]][:8]

    sub = df_in[cols].copy()
    out = pd.DataFrame(
        {
            "평균": sub.mean(numeric_only=True),
            "최대": sub.max(numeric_only=True),
            "최소": sub.min(numeric_only=True),
            "표준편차": sub.std(numeric_only=True),
        }
    ).T
    # 보기 좋게
    out = out.round(3)
    return out

proc_summary = build_process_summary(df.fillna(0))

# =========================
# Alarm table 만들기
# =========================
cols_all = df.columns.tolist()
sensor_cols = [c for c in cols_all if c != id_col and "label" not in c.lower() and "fail" not in c.lower() and "prob" not in c.lower()]

# 센서 컬럼은 stageN_ 형태만 우선 (알람 의미가 명확)
sensor_cols_stage = [c for c in sensor_cols if _STAGE_RE.match(c)]
if not sensor_cols_stage:
    sensor_cols_stage = sensor_cols[:60]  # fallback

df_for_alarm = df.copy()
for c in sensor_cols_stage:
    df_for_alarm[c] = safe_num(df_for_alarm[c])

limits_by_sensor = build_limits(df_for_alarm.fillna(0), sensor_cols_stage)
alarms = build_alarm_table(df_for_alarm.fillna(0), id_col, sensor_cols_stage, limits_by_sensor, max_rows=3000)

# =========================
# Main layout (좌 2개 영역 + 우 1개 알람영역)
# =========================
left_big, mid_big, right_big = st.columns([3.2, 2.4, 2.1], gap="small")

# -------- 좌: 게이지 + 공정 변수 테이블 + 공정 변수 트렌드(1개 선택)
with left_big:
    # 게이지
    st.markdown('<div class="panel-title">불량 예측 확률 (Gauge)</div>', unsafe_allow_html=True)

    p_color = "#a43434" if fail_prob >= 80 else "#f39c12" if fail_prob >= 51.48 else "#0d6e03"
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=fail_prob,
            number={"suffix": "%", "font": {"size": 30}, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0},
                "bar": {"color": p_color, "thickness": 0.65},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 51.48], "color": "#CBEAC4"},
                    {"range": [51.48, 80], "color": "#ffe8cb"},
                    {"range": [80, 100], "color": "#f5d2d2"},
                ],
            },
        )
    )
    fig_gauge.update_layout(height=H_GAUGE, margin=dict(t=10, b=0, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    # 공정 변수 테이블
    st.markdown('<div class="panel-title">공정 변수</div>', unsafe_allow_html=True)
    st.dataframe(proc_summary, use_container_width=True, height=H_TABLE)

    # 공정 변수 트렌드 (간단히: 선택한 컬럼 1개)
    st.markdown('<div class="panel-title">설비 시계열 데이터 (공정 변수 트렌드)</div>', unsafe_allow_html=True)
    trend_candidates = proc_summary.columns.tolist()
    if not trend_candidates:
        trend_candidates = ["prob"]
    trend_col = st.selectbox("변수 선택", options=trend_candidates, index=0, label_visibility="collapsed")

    # x축: id(있으면) / 없으면 index
    x_vals = df[id_col].astype(str) if id_col in df.columns else df.index.astype(str)
    y_vals = pd.to_numeric(df.get(trend_col, pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0)

    # 최근 구간만(너무 길면 렌더링 무거움)
    tail_n = min(60, len(df))
    x_plot = x_vals.tail(tail_n)
    y_plot = y_vals.tail(tail_n)

    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=x_plot,
            y=y_plot,
            mode="lines+markers",
            line=dict(color="#2d6cdf", width=2),
            marker=dict(size=6),
            hovertemplate=f"{trend_col}=%{{y:.4f}}<br>ID=%{{x}}<extra></extra>",
        )
    )
    fig_trend.update_layout(height=H_TREND, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    fig_trend.update_xaxes(tickangle=-45, tickfont=dict(size=10))
    st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

# -------- 중: 실시간 확률 라인 + anomaly score heatmap + 제품 불량 영향도
with mid_big:
    st.markdown('<div class="panel-title">실시간 불량 예측 확률 (최근 30)</div>', unsafe_allow_html=True)

    display_df = df.tail(30).copy()
    probs = display_df["prob"].to_numpy()
    n = len(probs)

    def rgba_by_prob(p, alpha):
        if p >= 80:
            return f"rgba(255,75,75,{alpha})"
        if p >= 51.48:
            return f"rgba(255,170,0,{alpha})"
        return f"rgba(0,204,102,{alpha})"

    marker_colors = [rgba_by_prob(float(p), 1.0 if i == n - 1 else 0.35) for i, p in enumerate(probs)]

    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=display_df[id_col].astype(str),
            y=display_df["prob"],
            mode="lines+markers",
            line=dict(color="#6e6e6e"),
            marker=dict(size=8, color=marker_colors),
            customdata=display_df[id_col].astype(str),
            hovertemplate="ID=%{customdata}<br>prob=%{y:.1f}%<extra></extra>",
        )
    )
    fig_line.add_hline(y=51.48, line_dash="dot", line_color="orange")
    fig_line.add_hline(y=80, line_dash="dash", line_color="red")
    fig_line.update_layout(height=H_LINE, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    fig_line.update_xaxes(tickangle=-45, tickfont=dict(size=10))
    fig_line.update_yaxes(range=[0, 105])
    st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})

    # anomaly score heatmap (예시: 최신 8개 제품의 일부 변수 Z-score 기반)
    st.markdown('<div class="panel-title">이상탐지 결과 (요약 Heatmap)</div>', unsafe_allow_html=True)
    # heatmap용 변수 선정: stage 기반 센서 중 일부 suffix만
    heat_sensors = []
    for suf in ["temp", "humidity", "flow_deviation", "viscosity_deviation", "co2_deviation"]:
        for i in range(1, 6):
            c = f"stage{i}_{suf}"
            if c in df.columns:
                heat_sensors.append(c)
                break
    if not heat_sensors:
        heat_sensors = sensor_cols_stage[:5]

    heat_df = df[heat_sensors].tail(8).copy().fillna(0)
    # 간단 z-score
    mu = df[heat_sensors].mean(numeric_only=True).fillna(0)
    sd = df[heat_sensors].std(numeric_only=True).replace(0, 1).fillna(1)
    z = (heat_df - mu) / sd
    z = z.abs().clip(0, 12)

    y_labels = [translate_feature(c) for c in heat_sensors]
    x_labels = [f"#{i+1}" for i in range(len(z))]

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=z.T.values,
            x=x_labels,
            y=y_labels,
            zmin=0,
            zmax=12,
            hovertemplate="항목=%{y}<br>샘플=%{x}<br>score=%{z:.2f}<extra></extra>",
        )
    )
    fig_heat.update_layout(height=H_HEAT, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    # 제품 불량 영향도 (최신 SHAP top 6)
    st.markdown('<div class="panel-title">제품 불량 영향도 (최신 Top)</div>', unsafe_allow_html=True)
    shap_df = pd.DataFrame({"eng": feature_names, "val": shap_vals_latest})
    shap_df["abs"] = shap_df["val"].abs()
    shap_df["항목"] = shap_df["eng"].apply(translate_feature)
    shap_top = shap_df.sort_values("abs", ascending=False).head(6).iloc[::-1]

    fig_bar = go.Figure(
        go.Bar(
            x=shap_top["abs"],
            y=shap_top["항목"],
            orientation="h",
            hovertemplate="%{y}<br>impact=%{x:.4f}<extra></extra>",
        )
    )
    fig_bar.update_layout(height=H_BAR, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

# -------- 우: 이상탐지 알람(선택 가능)
with right_big:
    st.markdown('<div class="panel-title">이상탐지 알람</div>', unsafe_allow_html=True)

    # 알람이 너무 많으면 렌더 부담이 커서 최근 N건만
    show_n = min(200, len(alarms))
    alarms_view = alarms.head(show_n).copy() if not alarms.empty else alarms

    if alarms_view.empty:
        st.info("알람이 없습니다.")
    else:
        grid_df = alarms_view[["event_id", "제품 ID", "공정 단계", "측정 항목", "sensor_raw", "order", "이탈폭", "측정값"]].copy()

        gb = GridOptionsBuilder.from_dataframe(grid_df)
        gb.configure_default_column(flex=1, minWidth=95, resizable=True, sortable=True, filter=True)
        gb.configure_selection(selection_mode="single", use_checkbox=False)
        gb.configure_column("event_id", hide=True)
        gb.configure_column("sensor_raw", hide=True)
        gb.configure_column("order", hide=True)
        gb.configure_column("측정값", type=["numericColumn"], valueFormatter="x.toFixed(3)")
        gb.configure_grid_options(domLayout="normal")
        grid_options = gb.build()

        grid = AgGrid(
            grid_df,
            gridOptions=grid_options,
            height=H_ALARM,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            fit_columns_on_grid_load=True,
            theme="streamlit",
            key="alarm_grid_onepage",
        )

        # 선택된 알람의 간단 요약(표 아래에 1줄만)
        selected_rows = grid.get("selected_rows", [])
        first = selected_rows[0] if isinstance(selected_rows, list) and selected_rows else None
        if first:
            st.caption(
                f"선택: ID={first.get('제품 ID')} / {first.get('공정 단계')} / {first.get('측정 항목')} / 이탈폭={first.get('이탈폭')}"
            )

# =========================
# (옵션) 화면이 "잘림"이 보이면:
# - 위 H_* 상수들을 더 줄이거나
# - KPI height / chart height를 줄이면 됨.
# =========================