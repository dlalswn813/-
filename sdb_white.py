import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import re
import time

# =========================================================
# Process Monitor (Latest) + Control-Limit Anomaly
# - 사후 분석용 메인 대시보드 (1페이지)
# - Stage Snapshot: 위험도 순 자동 정렬 (Priority Queue 완벽 구현)
# - 알람 개별 분리 & 버튼 증발 버그 완벽 해결 (과거 미조치 내역 추적)
# - Recent Trend 오토 포커스(Auto-Focus) & 연쇄 처리완료 완벽 연동
# - 5대 KPI 재배치 (OEE - 미조치 설비이상 - 총생산 - 불량수 - 불량률)
# - 실시간 데이터 시뮬레이터 (9초 단위 확정적 알람 주입, 변동성 극대화)
# =========================================================

st.set_page_config(page_title="공정 실시간 모니터링", layout="wide", initial_sidebar_state="expanded")

# ----------------------------
# Session State 초기화 (데이터 및 조치 상태)
# ----------------------------
if "resolved_alarms" not in st.session_state:
    st.session_state.resolved_alarms = set()
if "force_auto_target" not in st.session_state:
    st.session_state.force_auto_target = False

DATA_PATH = "mice_final_data_with_id.csv"
CTRL_K = 3.0
RECENT_WINDOW_N = 120

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

if "df" not in st.session_state:
    raw_df = load_csv(DATA_PATH)
    if "id" not in raw_df.columns or "label" not in raw_df.columns:
        st.error("CSV에 'id', 'label' 컬럼이 필요합니다.")
        st.stop()
    
    raw_df = raw_df.copy()
    id_num = pd.to_numeric(raw_df["id"], errors="coerce")
    raw_df["_id_sort"] = id_num if id_num.notna().mean() > 0.8 else raw_df["id"].astype(str)
    raw_df = raw_df.sort_values("_id_sort").reset_index(drop=True)
    raw_df["run"] = np.arange(len(raw_df))
    st.session_state.df = raw_df

df = st.session_state.df

FEATURE_COLS = [c for c in df.columns if c not in ("id", "label", "_id_sort", "run")]

STAGE_NAMES = {
    1: ("WS-01", "Wet Strip"),
    2: ("RR-02", "Residue Removal"),
    3: ("RN-03", "Rinse"),
    4: ("FWC-04", "Final Wet Cleaning"),
    5: ("FRD-05", "Final Rinse & Dry")
}

AXIS_SPEC = {
    "temp": (25, 35, 30),
    "humidity": (60, 80, 70),
    "flow_deviation": (-10, 10, 0),
    "density_deviation": (-10, 10, 0),
    "viscosity_deviation": (-10, 10, 0),
    "co2_deviation": (-10, 10, 0),
    "o2_deviation": (-5, 5, 0),
    "n_deviation": (-5, 5, 0),
}

SENSOR_KO = {
    "temp": "온도", "humidity": "습도", 
    "flow_deviation": "유량 편차", "density_deviation": "밀도 편차", 
    "viscosity_deviation": "점도 편차", "o2_deviation": "O2 편차", 
    "n_deviation": "N2 편차", "co2_deviation": "CO2 편차",
    "run": "생산 순번"
}

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
            stage_num = int(parts[0][-1])
            eq_code = STAGE_NAMES.get(stage_num, (f"ST-{stage_num}", ""))[0]
            metric = SENSOR_KO.get(parts[1], parts[1])
            return f"[{eq_code}] {metric}"
            
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

if "trend_stage" not in st.session_state:
    st.session_state.trend_stage = STAGES[0] if STAGES else 1
if "trend_metric" not in st.session_state:
    st.session_state.trend_metric = METRICS[0] if METRICS else "temp"

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
    "temp", "humidity", "flow_deviation", "density_deviation",
    "viscosity_deviation", "o2_deviation", "n_deviation", "co2_deviation",
]
METRICS = [m for m in CORE_METRICS_ORDER if m in METRICS] + [m for m in METRICS if m not in CORE_METRICS_ORDER]

# ----------------------------
# Styling
# ----------------------------
SAMSUNG_BLUE_2 = "#1428A0"
GOOD = "#10B981"
WARN = "#F59E0B"
BAD = "#DC2626"
RESOLVED_GRAY = "#9CA3AF"
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
      .kpi .v {{ font-size: 26px; font-weight: 900; color: var(--blue2); line-height: 1.05; }}
      .kpi .u {{ color: var(--muted); font-size: 13px; font-weight: 800; margin-left: 4px; }}

      .panel {{
        background: var(--panel);
        border: 1px solid var(--stroke);
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
      }}
      .pt {{ font-weight: 900; margin-bottom: 8px; }}
      
      .tooltip-container {{ position: relative; display: inline-block; cursor: help; }}
      .tooltip-container .tooltip-text {{
        visibility: hidden; width: 220px; background-color: rgba(0, 0, 0, 0.85); color: #fff;
        text-align: left; border-radius: 6px; padding: 10px; position: absolute;
        z-index: 100; bottom: 125%; left: 50%; margin-left: -110px; opacity: 0;
        transition: opacity 0.3s; font-size: 11px; line-height: 1.4; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
      }}
      .tooltip-container .tooltip-text::after {{
        content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px;
        border-width: 5px; border-style: solid; border-color: rgba(0, 0, 0, 0.85) transparent transparent transparent;
      }}
      .tooltip-container:hover .tooltip-text {{ visibility: visible; opacity: 1; }}
      .info-badge {{
        display: inline-block; width: 14px; height: 14px; line-height: 14px; border-radius: 50%;
        background-color: rgba(255,255,255,0.3); color: white; font-size: 10px; text-align: center;
        margin-left: 5px; vertical-align: middle;
      }}

      div[data-testid="stControlItem"] div[role="group"] {{
          width: 100% !important; padding: 8px 0 !important; display: flex !important;
          justify-content: space-between !important; align-items: center !important;
          margin-bottom: 5px !important; border-bottom: 1px solid #E5E7EB; 
      }}
      div[data-testid="stControlItem"] label[data-testid="stWidgetLabel"] p {{
          font-size: 15px !important; font-weight: 700 !important; color: #334155 !important;
      }}
      div[data-testid="stControlItem"] input[type="checkbox"][role="switch"] {{
          transform: scale(1.4) !important; margin-right: 15px !important; cursor: pointer !important;
      }}

    </style>
    """,
    unsafe_allow_html=True,
)

def kpi(container, label, value, unit="", fmt="{:.2f}"):
    v = "-" if value is None or (isinstance(value, float) and np.isnan(value)) else fmt.format(value)
    html_str = f'<div class="kpi"><div class="l">{label}</div><div><span class="v">{v}</span><span class="u">{unit}</span></div></div>'
    container.markdown(html_str, unsafe_allow_html=True)

def fmtv(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{float(x):.{nd}f}"

# ----------------------------
# Latest sample + recent window
# ----------------------------
t = int(df["run"].iloc[-1])
cur = df.iloc[-1]
run_val = int(cur["run"])

w0 = max(0, t - RECENT_WINDOW_N + 1)
hist = df.iloc[w0 : t + 1].copy()

# ----------------------------
# Control limits & Anomaly Extraction
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

def limit_exceed_score(val, feature: str) -> float:
    if feature not in CTRL_LIMITS: return 0.0
    mu, sd, lcl, ucl = CTRL_LIMITS[feature]
    if val is None or (isinstance(val, float) and np.isnan(val)) or (not np.isfinite(mu)) or (not np.isfinite(sd)) or sd <= 0:
        return 0.0
    return abs((float(val) - mu) / sd)

# 상세 알람 리스트 추출 (최근 120건 내 발생한 모든 알람)
anomaly_list = []
for _, r in hist.iterrows():
    r_val = int(r["run"])
    product_id = r.get("id", "-")
    for c in FEATURE_COLS:
        v = pd.to_numeric(r.get(c), errors="coerce")
        z = limit_exceed_score(v, c)
        if z >= 3.0: 
            feature_ko = translate_feature_name(c) 
            anomaly_list.append({
                "uid": f"alarm_{r_val}_{c}", 
                "run": r_val,
                "id": product_id,
                "details": f"{feature_ko}: {v:.2f}",
                "severity": z
            })

anomaly_list.sort(key=lambda x: x["severity"], reverse=True)
active_alarms = [a for a in anomaly_list if a["uid"] not in st.session_state.resolved_alarms]
resolved_alarms = [a for a in anomaly_list if a["uid"] in st.session_state.resolved_alarms]
active_anom_count = len(active_alarms)

# ----------------------------
# 🚀 오토 포커스 (Auto-Targeting) & 버튼 동기화 핵심 로직
# ----------------------------
# 1. 처리완료를 누른 직후라면, 미조치 알람 중 가장 위험한 곳으로 강제 이동
if st.session_state.get("force_auto_target", False):
    if active_alarms:
        worst_uid = active_alarms[0]["uid"]
        # uid 형태: alarm_16900_stage2_temp -> 분리하여 필터용 변수 획득
        fcol_str = worst_uid.split("_", 2)[2] 
        stage_part, metric_part = fcol_str.split("_", 1)
        st.session_state.trend_stage = int(stage_part.replace("stage", ""))
        st.session_state.trend_metric = metric_part
    st.session_state.force_auto_target = False

# ----------------------------
# 카드용 데이터 구조 (우선순위 자동 정렬)
# ----------------------------
stage_data = []

for s in STAGES:
    worst_unresolved_z, worst_unresolved_m, worst_unresolved_v = -1, None, np.nan
    worst_resolved_z, worst_resolved_m, worst_resolved_v = -1, None, np.nan
    worst_normal_z, worst_normal_m, worst_normal_v = -1, METRICS[0], np.nan
    
    for m in METRICS:
        f = fcol(s, m)
        if f in df.columns:
            v = pd.to_numeric(cur.get(f), errors="coerce")
            z = limit_exceed_score(v, f)
            uid = f"alarm_{run_val}_{f}"
            
            if z >= 3.0:
                if uid in st.session_state.resolved_alarms:
                    if z > worst_resolved_z:
                        worst_resolved_z, worst_resolved_m, worst_resolved_v = z, m, v
                else:
                    if z > worst_unresolved_z:
                        worst_unresolved_z, worst_unresolved_m, worst_unresolved_v = z, m, v
            else:
                if z > worst_normal_z:
                    worst_normal_z, worst_normal_m, worst_normal_v = z, m, v

    if worst_unresolved_z >= 3.0:
        stage_data.append({"stage": s, "metric": worst_unresolved_m, "val": worst_unresolved_v, "z": worst_unresolved_z, "state": "active"})
    elif worst_resolved_z >= 3.0:
        stage_data.append({"stage": s, "metric": worst_resolved_m, "val": worst_resolved_v, "z": worst_resolved_z, "state": "resolved"})
    else:
        stage_data.append({"stage": s, "metric": worst_normal_m, "val": worst_normal_v, "z": worst_normal_z, "state": "normal"})

def state_priority(state):
    if state == "active": return 3
    if state == "resolved": return 2
    return 1

stage_data.sort(key=lambda x: (state_priority(x["state"]), x["z"]), reverse=True)

# ----------------------------
# KPI 데이터 준비 및 OEE
# ----------------------------
total_prod = int(len(df))
defect_count = int((df["label"] == 1).sum())
defect_rate = (defect_count / total_prod) if total_prod > 0 else np.nan

base_oee = 99.5
quality_loss = (defect_rate * 100) if np.isfinite(defect_rate) else 0
performance_loss = active_anom_count * 0.15 
oee_value = max(0.0, base_oee - quality_loss - performance_loss)

# ----------------------------
# Sidebar: Remote Control & Simulation
# ----------------------------
with st.sidebar:
    st.markdown('<div class="panel" style="margin-top: 10px; margin-bottom: 20px;"><div class="pt" style="font-size:14px;">Remote Control <br><span style="display:inline-block; margin-top:4px; padding:2px 10px; border-radius:999px; background:#EEF2FF; color:#4338CA; font-size:12px; font-weight:800; border:1px solid rgba(67,56,202,0.2);">설비 원격 제어</span></div></div>', unsafe_allow_html=True)
    
    st.toggle("냉각기 (Chiller)", value=True, key="c_chiller")
    st.toggle("가습기 (Humidifier)", value=True, key="c_humid")
    st.toggle("메인 펌프 (Pump)", value=True, key="c_pump")
    st.toggle("진공 배기 (Vacuum)", value=False, key="c_vac")
    st.toggle("N2 퍼지 밸브", value=True, key="c_n2")
    st.toggle("CO2 버블러", value=True, key="c_co2")
    st.toggle("희석수 밸브", value=False, key="c_diw")
    
    st.markdown('<div style="margin: 15px 0;"></div>', unsafe_allow_html=True)
    st.toggle("비상 정지 (EMG)", value=False, key="c_emg")

    if st.session_state.c_emg:
        st.markdown('<div style="background:#FEE2E2; color:#DC2626; border:1px solid #F87171; border-radius:8px; padding:12px; font-weight:900; font-size:15px; margin-top:15px; text-align:center;">[비상] 전 라인 가동 중단됨</div>', unsafe_allow_html=True)

    st.markdown('<hr style="margin: 20px 0; border-color: #E5E7EB;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:14px; font-weight:900; color:#111827; margin-bottom:10px;">시뮬레이터</div>', unsafe_allow_html=True)
    sim_on = st.toggle("실시간 데이터 갱신 (3초)", value=False, key="sim_toggle")

# ----------------------------
# Top bar 
# ----------------------------
top_l, top_r = st.columns([1.6, 1.4])
with top_l:
    st.markdown('<div class="topbar" style="height: 76px;"><div class="brand">SPARTA ELECTRONICS · 공정 모니터링</div><div class="subtitle">Wet PR Strip & Clean 품질 제어 · 최신 데이터 뷰</div></div>', unsafe_allow_html=True)

with top_r:
    tr1, tr2 = st.columns([3.2, 1.0])
    with tr1:
        st.markdown(f'<div class="topbar" style="height: 76px;"><div class="rightmeta" style="font-size:18px;">현재 생산 제품 현황</div><div class="subtitle rightmeta">제품 ID={cur["id"]} · 분석 구간=최근 {len(hist)}건</div></div>', unsafe_allow_html=True)
    with tr2:
        popover_bg = "#FEF2F2" if active_anom_count > 0 else "#F3F4F6"
        popover_color = "#DC2626" if active_anom_count > 0 else "#6B7280"
        popover_border = "#F87171" if active_anom_count > 0 else "#E5E7EB"
        
        st.markdown(f'''
        <style>
        div[data-testid="stPopover"] button {{
            height: 76px; border-radius: 14px; border: 1px solid {popover_border}; 
            background-color: {popover_bg}; color: {popover_color}; font-weight: 900; 
            font-size: 16px; width: 100%;
        }}
        </style>
        ''', unsafe_allow_html=True)
        
        with st.popover(f"🔔 미조치 알람: {active_anom_count}건", use_container_width=True):
            st.markdown("#### 설비 이상 조치 현황")
            if not active_alarms and not resolved_alarms:
                st.info("최근 감지된 설비 이상이 없습니다.")
            
            for a in active_alarms:
                with st.container():
                    c_text, c_btn = st.columns([3, 1])
                    with c_text:
                        st.error(f"[위험도: {a['severity']:.1f}σ] 순번 {a['run']}\n\n{a['details']}")
                    with c_btn:
                        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True) 
                        if st.button("처리완료", key=f"btn_pop_{a['uid']}", use_container_width=True):
                            st.session_state.resolved_alarms.add(a['uid'])
                            st.session_state.force_auto_target = True
                            st.rerun()
            
            if resolved_alarms:
                st.markdown("<hr style='margin: 15px 0 10px 0; border-color: #E5E7EB;'>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:13px; color:#6B7280; font-weight:700; margin-bottom:10px;'>과거 조치 로그 (History)</div>", unsafe_allow_html=True)
                for a in resolved_alarms:
                    st.markdown(f'''
                    <div style="background-color: #F9FAFB; color: #9CA3AF; padding: 10px; border-radius: 8px; margin-bottom: 8px; border: 1px solid #E5E7EB; font-size: 14px;">
                        <del><b>[조치완료] 순번 {a['run']} (ID: {a['id']})</b><br>{a['details']}</del>
                    </div>
                    ''', unsafe_allow_html=True)

# ----------------------------
# KPI row 
# ----------------------------
k1, k2, k3, k4, k5 = st.columns(5)
kpi(k1, "설비효율 (OEE)", float(oee_value), "%", "{:.1f}")
kpi(k2, "미조치 설비 이상", float(active_anom_count), "건", "{:.0f}")
kpi(k3, "총 생산량", float(total_prod), "개", "{:.0f}")
kpi(k4, "불량 제품 수", float(defect_count), "개", "{:.0f}")
kpi(k5, "불량 비율", defect_rate * 100 if np.isfinite(defect_rate) else np.nan, "%", "{:.2f}")

st.markdown("")

# ----------------------------
# Main layout (카드 렌더링)
# ----------------------------
left, right = st.columns([2.55, 1.45], gap="medium")

with left:
    st.markdown('<div class="panel"><div class="pt">공정별 실시간 상태 요약 (SCADA View) <span class="pill" style="margin-left:8px; display:inline-block; padding:2px 10px; border-radius:999px; background:rgba(220,38,38,0.1); color:#DC2626; font-size:12px; font-weight:800; border:1px solid rgba(220,38,38,0.3);">우선순위 자동정렬</span></div></div>', unsafe_allow_html=True)

    cols = st.columns(len(stage_data), gap="small")

    for i, s_info in enumerate(stage_data):
        s = s_info["stage"]
        target_m = s_info["metric"]
        target_f = fcol(s, target_m)
        val = s_info["val"]
        state = s_info["state"]
        z_score = s_info["z"]
        
        metric_ko = SENSOR_KO.get(target_m, target_m)
        eq_code, eq_desc = STAGE_NAMES.get(s, (f"ST-{s}", f"공정 {s}"))
        
        mu, sd, lcl, ucl = CTRL_LIMITS.get(target_f, (0, 0, 0, 0))
        
        if state == "active":
            bg_color = BAD if z_score >= 3.5 else WARN
        elif state == "resolved":
            bg_color = RESOLVED_GRAY
        else:
            bg_color = GOOD
            
        val_str = f"{z_score:.1f} σ" if pd.notna(z_score) and z_score >= 0 else "-"
        val_display = f"{val:.2f}" if pd.notna(val) else "-"

        recent_10 = hist[target_f].tail(10).values
        svg_line = ""
        if len(recent_10) > 1:
            plot_min, plot_max = (lcl, ucl) if np.isfinite(lcl) and np.isfinite(ucl) else (np.nanmin(recent_10), np.nanmax(recent_10))
            if plot_max == plot_min: 
                plot_min, plot_max = plot_min - 1, plot_max + 1
            
            points = []
            n_vals = len(recent_10)
            for j, v in enumerate(recent_10):
                if pd.isna(v): continue
                x = (j / (n_vals - 1)) * 100
                y = 30 - (((v - plot_min) / (plot_max - plot_min)) * 30)
                y = max(0, min(30, y))
                points.append(f"{x},{y}")
            
            polyline_points = " ".join(points)
            svg_line = f'<polyline points="{polyline_points}" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />'
            
            last_x, last_y = points[-1].split(",")
            svg_line += f'<circle cx="{last_x}" cy="{last_y}" r="2.5" fill="#FCD34D" stroke="white" stroke-width="0.5" />'
            
            mu_y = 30 - (((mu - plot_min) / (plot_max - plot_min)) * 30)
            if 0 <= mu_y <= 30:
                svg_line += f'<line x1="0" y1="{mu_y}" x2="100" y2="{mu_y}" stroke="rgba(255,255,255,0.4)" stroke-width="1" stroke-dasharray="2,2" />'

        html_card = (
            f'<div style="background-color: {bg_color}; border-radius: 12px; padding: 15px 15px 10px 15px; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 210px; display: flex; flex-direction: column; justify-content: space-between;">'
            f'<div><div style="font-weight: 900; font-size: 16px; margin-bottom: 0px; letter-spacing: 0.5px;">{eq_code} <span class="tooltip-container"><span class="info-badge">i</span><span class="tooltip-text"><b>[시그마 이탈도 가이드]</b><br>정상 범위(평균)에서 얼마나 벗어났는지 나타냅니다.<br><br>🟢 정상: 0.0 ~ 3.0 σ<br>🟡 주의: 3.0 ~ 3.5 σ<br>🔴 <b>위험: 3.5 σ 초과</b></span></span></div>'
            f'<div style="font-size: 11px; opacity: 0.8; margin-bottom: 4px;">{eq_desc}</div>'
            f'<div style="font-size: 13px; font-weight: 700; opacity: 1.0;">{metric_ko}</div></div>'
            f'<div style="flex-grow: 1; display: flex; align-items: center; justify-content: center;"><span style="font-size: 42px; font-weight: 900; letter-spacing: -1px; text-shadow: 1px 1px 2px rgba(0,0,0,0.2); border-bottom: 1px dashed rgba(255,255,255,0.4);">{val_str}</span></div>'
            f'<div style="width: 100%; text-align: left;"><div style="font-size: 10px; color: rgba(255,255,255,0.8); margin-bottom: 2px;">최근 10건 추이</div>'
            f'<div style="height: 35px; width: 100%;"><svg width="100%" height="100%" viewBox="0 0 100 30" preserveAspectRatio="none" style="overflow: visible;">{svg_line}</svg></div></div></div>'
        )

        with cols[i]:
            st.markdown(html_card, unsafe_allow_html=True)

    st.markdown("")

    # ----------------------------
    # Recent Trend (Auto-Target & 버그 없는 버튼 렌더링)
    # ----------------------------
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        st.selectbox("분석 공정 (설비) 선택", STAGES, key="trend_stage", format_func=lambda x: f"{STAGE_NAMES.get(x, (f'ST-{x}', ''))[0]} ({STAGE_NAMES.get(x, ('', f'공정 {x}'))[1]})")
    with filter_col2:
        st.selectbox("분석 지표 선택", METRICS, key="trend_metric", format_func=lambda x: SENSOR_KO.get(x, x))
    
    trend_feature = fcol(st.session_state.trend_stage, st.session_state.trend_metric)
    
    # 🚀 현재 선택된 지표에 대해 '미조치 알람'이 단 하나라도 남아있는지 철저히 스캔
    feature_active_uids = [a["uid"] for a in active_alarms if a["uid"].endswith(f"_{trend_feature}")]
    is_active_in_trend = len(feature_active_uids) > 0

    rc_l, rc_r = st.columns([4, 1])
    with rc_l:
        st.markdown('<div class="pt" style="font-size:15px; padding-top:6px;">Recent Trend <span class="pill" style="margin-left:8px; display:inline-block; padding:2px 10px; border-radius:999px; background:rgba(20,40,160,0.05); color:#1428A0; font-size:12px; font-weight:800; border:1px solid rgba(20,40,160,0.2);">집중 분석창</span></div>', unsafe_allow_html=True)
    with rc_r:
        if is_active_in_trend:
            st.markdown("""
            <style>
            div[data-testid="stButton"] button {
                background-color: #10B981; color: white; border: None; font-weight: 800; height: 38px;
            }
            div[data-testid="stButton"] button:hover { background-color: #059669; color: white; }
            </style>""", unsafe_allow_html=True)
            
            # 버튼 클릭 시 해당 지표의 '모든' 미조치 알람을 한 번에 해결하고 다음 타겟으로 이동
            if st.button("현재 지표 처리완료", key="btn_trend_resolve", use_container_width=True):
                for uid in feature_active_uids:
                    st.session_state.resolved_alarms.add(uid)
                st.session_state.force_auto_target = True 
                st.rerun()

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
        height=330,
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
    # Stage Profile 
    # ----------------------------
    st.markdown('<div class="panel"><div class="pt">Stage Profile <span class="pill" style="margin-left:8px; display:inline-block; padding:2px 10px; border-radius:999px; background:rgba(20,40,160,0.05); color:#1428A0; font-size:12px; font-weight:800; border:1px solid rgba(20,40,160,0.2);">8개 지표 현황</span></div></div>', unsafe_allow_html=True)

    MINI_METRICS = [
        "temp", "humidity", "flow_deviation", "density_deviation",
        "viscosity_deviation", "o2_deviation", "n_deviation", "co2_deviation",
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
        
        y_min, y_max, target_val = AXIS_SPEC.get(m, (None, None, None))

        fig_m = go.Figure()
        x_labels = [STAGE_NAMES.get(s, (f"ST-{s}", ""))[0] for s in xs]
        
        fig_m.add_trace(go.Scatter(x=x_labels, y=ys, mode="lines+markers", name=metric_ko, line=dict(color=SAMSUNG_BLUE_2, width=2)))
        
        if target_val is not None:
            fig_m.add_hline(y=target_val, line_dash="dash", line_color="#10B981", line_width=1.5, opacity=0.7)

        layout_dict = dict(
            height=170,
            margin=dict(l=5, r=5, t=25, b=5),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, size=10),
            xaxis=dict(tickmode="array", tickvals=x_labels, gridcolor="rgba(0,0,0,0.05)", title_font=dict(size=9)),
            showlegend=False,
            title=dict(text=metric_ko, x=0, y=1, font_size=12),
        )
        
        if y_min is not None and y_max is not None:
            layout_dict["yaxis"] = dict(range=[y_min, y_max], gridcolor="rgba(0,0,0,0.05)", zeroline=False)
        else:
            layout_dict["yaxis"] = dict(gridcolor="rgba(0,0,0,0.05)")

        fig_m.update_layout(**layout_dict)
        figs.append(fig_m)

    for idx, fig_m in enumerate(figs):
        with grid_cols[idx % 4]:
            st.plotly_chart(fig_m, use_container_width=True)

# ----------------------------
# 🚀 실시간 데이터 시뮬레이터 (다이내믹 변동 로직)
# ----------------------------
if st.session_state.get("sim_toggle", False):
    time.sleep(3)
    
    last_row = df.iloc[-1].copy()
    new_row = last_row.copy()
    
    new_run = int(last_row['run']) + 1
    new_row['run'] = new_run
    new_row['id'] = f"PRD-SIM-{new_run:04d}" 
    new_row['label'] = 0 
    
    # 3사이클(약 9초)마다 100% 확률로 하나의 랜덤 센서를 확정적으로 폭주시킴
    force_spike_feature = None
    if new_run % 3 == 0:
        force_spike_feature = np.random.choice(FEATURE_COLS)
    
    for c in FEATURE_COLS:
        v = pd.to_numeric(last_row.get(c), errors="coerce")
        if pd.notna(v):
            mu, sd, lcl, ucl = CTRL_LIMITS.get(c, (v, 1.0, v-3, v+3))
            
            # 이전보다 노이즈 폭을 넓혀서 생동감 부여
            noise = np.random.normal(0, max(sd * 0.6, 0.1))
            pull = (mu - v) * 0.1 
            
            spike = 0
            if c == force_spike_feature:
                # 강제 폭주 트리거 발생 (4.0 ~ 7.0 시그마 밖으로 튕겨냄)
                spike = np.random.choice([1, -1]) * np.random.uniform(4.0, 7.0) * max(sd, 0.5)
            
            new_row[c] = v + pull + noise + spike
            
    st.session_state.df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    st.rerun()
