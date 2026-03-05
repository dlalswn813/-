import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# CatBoost (model + SHAP)
from catboost import CatBoostClassifier, Pool

# =========================================================
# Process Monitor (Latest) + Model + Control-Limit Anomaly
# - NO sidebar / NO theme switch / NO operator / NO autoplay
# - Show ONLY the last sample (latest product)
# - ALARM banner uses model probability -> 정상/주의/위험
# - KPI 4개: 총 생산량 / 불량 제품 수 / 불량 비율 / 설비 이상 감지 수
# - Stage Snapshot: Stage별 4변수(온도/습도 + deviation 2개) 표시 + 이상 stage는 빨간점
# - Recent Trend: "가장 한계선 초과가 큰" feature 자동 선택하여 표시 (y축 자동 변경)
# - Right: (1) 제품 불량 영향도(현재 제품 SHAP top5) 가로 막대
#         (2) Stage1~5 기준 8개 미니 그래프(온도/습도/편차들)
# =========================================================

st.set_page_config(page_title="Process Monitor (Latest)", layout="wide")

DATA_PATH = "mice_final_data_with_id.csv"
MODEL_PATH = "catboost_final_model.cbm"

# 모델 threshold (요청: "모델이 잡은 Threshold")
# 학습 시 최적 threshold가 따로 저장되어 있으면 여기에 넣으면 됨.
MODEL_WARN_THRESHOLD = 0.50
MODEL_RISK_THRESHOLD = 0.80

# 관리도 기반 한계선(기본): mean ± k*sigma
CTRL_K = 3.0
RECENT_WINDOW_N = 120

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_model(path: str) -> CatBoostClassifier:
    m = CatBoostClassifier()
    m.load_model(path)
    return m

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
# Styling (fixed)
# ----------------------------
SAMSUNG_BLUE_2 = "#1E40FF"
GOOD = "#22C55E"
WARN = "#F59E0B"
BAD = "#EF4444"
TEXT = "#E9EDF6"
MUTED = "#AAB6D3"

BG = "#05070D"
PANEL = "#0F1628"
STROKE = "#1E2942"

st.markdown(
    f"""
    <style>
      :root {{
        --bg:{BG}; --panel:{PANEL}; --stroke:{STROKE};
        --text:{TEXT}; --muted:{MUTED};
        --blue2:{SAMSUNG_BLUE_2};
        --good:{GOOD}; --warn:{WARN}; --bad:{BAD};
      }}
      .stApp {{ background: var(--bg); color: var(--text); }}
      header, [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
      footer {{ visibility: hidden; }}

      /* hide sidebar completely */
      [data-testid="stSidebar"], section[data-testid="stSidebar"] {{ display: none !important; }}

      .topbar {{
        background: linear-gradient(180deg, rgba(20,40,160,0.22), rgba(17,24,41,0.95));
        border: 1px solid rgba(30,64,255,0.35);
        border-radius: 14px;
        padding: 12px 16px;
        margin-bottom: 10px;
      }}
      .brand {{ font-size: 24px; font-weight: 900; letter-spacing: 0.4px; color: var(--blue2); }}
      .subtitle {{ color: var(--muted); font-weight: 700; margin-top: 2px; }}
      .rightmeta {{ text-align: right; font-weight: 900; }}

      .banner {{
        border-radius: 14px;
        padding: 12px 16px;
        margin-bottom: 10px;
        border: 1px solid var(--stroke);
        background: rgba(255,255,255,0.03);
      }}
      .banner.good {{
        border-color: rgba(34,197,94,0.40);
        background: rgba(34,197,94,0.10);
      }}
      .banner.warn {{
        border-color: rgba(245,158,11,0.45);
        background: rgba(245,158,11,0.10);
      }}
      .banner.bad {{
        border-color: rgba(239,68,68,0.45);
        background: rgba(239,68,68,0.10);
      }}
      .banner .big {{ font-size: 18px; font-weight: 900; }}

      .kpi {{
        background: var(--panel);
        border: 1px solid var(--stroke);
        border-radius: 12px;
        padding: 12px 14px;
        height: 84px;
      }}
      .kpi .l {{ color: var(--muted); font-size: 12px; font-weight: 800; }}
      .kpi .v {{ font-size: 30px; font-weight: 900; color: var(--blue2); line-height: 1.05; }}
      .kpi .u {{ color: var(--muted); font-size: 13px; font-weight: 800; margin-left: 6px; }}

      .panel {{
        background: var(--panel);
        border: 1px solid var(--stroke);
        border-radius: 12px;
        padding: 12px 14px;
      }}
      .pt {{ font-weight: 900; margin-bottom: 8px; }}
      .muted {{ color: var(--muted); }}
      .pill {{
        display:inline-block; padding: 2px 10px; border-radius:999px;
        border: 1px solid rgba(30,64,255,0.35);
        background: rgba(20,40,160,0.12);
        color: var(--muted); font-size: 12px; font-weight: 800; margin-left: 8px;
      }}

      .stagecard {{
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--stroke);
        border-radius: 12px;
        padding: 10px 12px;
        min-height: 190px;
      }}
      .stagehdr {{
        display:flex; justify-content:space-between; align-items:center;
        margin-bottom: 8px;
      }}
      .stagetitle {{ font-weight: 900; }}
      .statusdot {{
        width:10px; height:10px; border-radius:999px; display:inline-block;
        margin-left: 8px;
      }}
      .row2 {{ display:flex; justify-content:space-between; gap:10px; }}
      .m {{
        flex:1;
        background: rgba(0,0,0,0.15);
        border: 1px solid rgba(37,48,74,0.9);
        border-radius: 10px;
        padding: 8px 10px;
      }}
      .ml {{ color: var(--muted); font-size: 11px; font-weight: 800; }}
      .mv {{ font-size: 18px; font-weight: 900; }}
      .small {{ font-size: 12px; }}
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
# - baseline: 최근 window (hist) 사용
# - feature별로 numeric 변환 후 mean/std
# ----------------------------
def compute_ctrl_limits(hist_df: pd.DataFrame, feature_cols: list[str], k: float) -> dict[str, tuple[float, float, float, float]]:
    # returns {feature: (mean, std, lcl, ucl)}
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
    # "한계선 얼마나 넘었는지"를 점수화: |z| (std 단위)
    if feature not in CTRL_LIMITS:
        return 0.0
    mu, sd, lcl, ucl = CTRL_LIMITS[feature]
    if val is None or (isinstance(val, float) and np.isnan(val)) or (not np.isfinite(mu)) or (not np.isfinite(sd)) or sd <= 0:
        return 0.0
    return abs((float(val) - mu) / sd)

# 설비 이상 감지(run 단위): 최근 window 내에서 "어떤 feature든 한계선 초과"면 1
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
# 파생변수 계산
# ----------------------------
import re

def _safe_num(x):
    return pd.to_numeric(x, errors="coerce")

def compute_derived_feature_for_row(row: pd.Series, feat_name: str, stages: list[int]) -> float:
    """
    Support patterns:
      - {metric}_diff_i_j  => stage{i}_{metric} - stage{j}_{metric}
      - {metric}_slope     => slope of y=stage{1..5}_{metric} vs x=stage index
    """
    m = re.match(r"^(?P<metric>.+)_diff_(?P<i>\d+)_(?P<j>\d+)$", feat_name)
    if m:
        metric = m.group("metric")
        i = int(m.group("i"))
        j = int(m.group("j"))
        c_i = f"stage{i}_{metric}"
        c_j = f"stage{j}_{metric}"
        if c_i in row.index and c_j in row.index:
            vi = _safe_num(row.get(c_i))
            vj = _safe_num(row.get(c_j))
            if pd.notna(vi) and pd.notna(vj):
                return float(vi - vj)
        return np.nan

    m = re.match(r"^(?P<metric>.+)_slope$", feat_name)
    if m:
        metric = m.group("metric")
        xs = []
        ys = []
        for s in stages:
            c = f"stage{s}_{metric}"
            if c in row.index:
                v = _safe_num(row.get(c))
                if pd.notna(v):
                    xs.append(float(s))
                    ys.append(float(v))
        if len(xs) >= 2:
            # linear fit slope
            slope = np.polyfit(xs, ys, 1)[0]
            return float(slope)
        return np.nan

    return np.nan  # unhandled pattern

# ----------------------------
# Model inference (probability) - with derived feature generation
# ----------------------------
X_cur = None
model = None
proba_ng = np.nan
model_ok = False
model_features = []
model_feature_importance = None
unhandled_missing = []

try:
    model = load_model(MODEL_PATH)
    # CatBoost feature names (version-safe)
    model_features = None

    # 1) many versions expose feature_names_
    if hasattr(model, "feature_names_") and model.feature_names_:
        model_features = list(model.feature_names_)

    # 2) some versions expose feature_names_ only after fitting; try model.get_param fallback not reliable
    if not model_features:
        # try reading from model metadata if available
        try:
            # Pool is not required; this is just a fallback attempt
            model_features = list(model._object._get_feature_names())  # type: ignore[attr-defined]
        except Exception:
            model_features = None

    if not model_features:
        raise RuntimeError("모델에서 feature name 목록을 읽을 수 없습니다. (catboost 버전/모델 메타 확인 필요)")

    # base row = last product
    base_row = df.iloc[-1].copy()

    # build a dict for X_cur columns
    xdict = {}

    for f in model_features:
        if f in df.columns:
            xdict[f] = _safe_num(base_row.get(f))
        else:
            # try derive
            v = compute_derived_feature_for_row(base_row, f, STAGES)
            xdict[f] = v
            # if still nan, mark as unhandled (or insufficient base columns)
            if not np.isfinite(v):
                unhandled_missing.append(f)

    # Create X_cur with exact order
    X_cur = pd.DataFrame([xdict], columns=model_features)

    # If there are missing features we couldn't compute, we stop model prediction
    if unhandled_missing:
        model_ok = False
        st.warning(
            "모델이 요구하는 피처 중 계산/생성이 안 된 항목이 있어 모델 예측을 비활성화합니다.\n"
            f"예시: {unhandled_missing[:8]}{' ...' if len(unhandled_missing) > 8 else ''}"
        )
    else:
        model_ok = True
        proba_ng = float(model.predict_proba(X_cur)[0][1])

    # Global importance (works regardless)
    fi = model.get_feature_importance(type="FeatureImportance")
    model_feature_importance = pd.DataFrame({"feature": model_features, "importance": fi})

except Exception as e:
    model = None
    model_ok = False
    proba_ng = np.nan
    model_feature_importance = None
    st.error(f"모델 로드/추론 실패: {e}")

# ----------------------------
# Risk level (ALWAYS define: fixes NameError)
# ----------------------------
def risk_level(p: float):
    if not np.isfinite(p):
        return ("-", "warn", WARN)
    if p >= MODEL_RISK_THRESHOLD:
        return ("위험", "bad", BAD)
    if p >= MODEL_WARN_THRESHOLD:
        return ("주의", "warn", WARN)
    return ("정상", "good", GOOD)

level_txt, banner_class, banner_color = risk_level(proba_ng)

if not model_ok:
    level_txt = "모델 OFF"
    banner_class = "warn"
    banner_color = WARN

# ----------------------------
# KPI (요청한 4개로 교체)
# ----------------------------
total_prod = int(len(df))
defect_count = int((df["label"] == 1).sum())
defect_rate = (defect_count / total_prod) if total_prod > 0 else np.nan
equip_anom_count = int(equip_anom_recent)  # 최근 window 기준 (원하면 전체로 바꾸면 됨)

# ----------------------------
# Stage Snapshot: 4 variables per stage + anomaly dot
# - 온도/습도 + (deviation 2개는 현재 제품 기준 한계선 초과 점수 top2)
# ----------------------------
def pick_stage_dev_metrics(stage: int, topk: int = 2) -> list[str]:
    candidates = []
    for m in DEV_METRICS:
        f = fcol(stage, m)
        if f in df.columns:
            v = pd.to_numeric(cur.get(f), errors="coerce")
            score = limit_exceed_score(v, f)
            over = is_over_limit(v, f)
            # 초과된 것 우선 + 점수 큰 것 우선
            candidates.append(((10.0 if over else 0.0) + score, m))
    candidates.sort(reverse=True)
    picked = [m for _, m in candidates[:topk]]
    # deviation이 부족하면 CORE 순으로 채움
    if len(picked) < topk:
        for m in CORE_METRICS_ORDER:
            if m in DEV_METRICS and m not in picked:
                f = fcol(stage, m)
                if f in df.columns:
                    picked.append(m)
            if len(picked) >= topk:
                break
    return picked

# stage anomaly: temp/humidity + 선택된 deviation 2개 중 하나라도 한계선 초과
def stage_is_anomaly(stage: int) -> bool:
    feats = []
    for m in ["temp", "humidity"]:
        f = fcol(stage, m)
        if f in df.columns:
            feats.append(f)
    for m in pick_stage_dev_metrics(stage, 2):
        f = fcol(stage, m)
        if f in df.columns:
            feats.append(f)

    for f in feats:
        v = pd.to_numeric(cur.get(f), errors="coerce")
        if is_over_limit(v, f):
            return True
    return False

# ----------------------------
# Recent Trend: "설비 이상 중 가장 위험한" feature 자동 선택
# - 최근 window 기준 ctrl limits 대비 현재 제품 z-score max feature
# - 초과한 feature가 있으면 그 중 max, 없으면 전체 중 max
# ----------------------------
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

trend_feature = pick_most_critical_feature()

# ----------------------------
# SHAP 영향도 (현재 제품) top5
# ----------------------------
shap_top5 = None
try:
    if (model is not None) and model_ok and (X_cur is not None):
        pool_cur = Pool(X_cur)
        shap_vals = model.get_feature_importance(type="ShapValues", data=pool_cur)
        # shap_vals shape: (1, n_features+1) ; 마지막 컬럼은 expected value
        shap_vec = np.array(shap_vals)[0][:-1]
        imp = pd.DataFrame(
            {
                "feature": model_features,
                "shap": shap_vec,
                "abs_shap": np.abs(shap_vec),
            }
        ).sort_values("abs_shap", ascending=False)
        shap_top5 = imp.head(5).copy()
except Exception as e:
    shap_top5 = None
    st.error(f"SHAP 계산 실패: {e}")

# ----------------------------
# Top bar + banner (모델 확률 기반)
# ----------------------------
top_l, top_c, top_r = st.columns([1.7, 2.0, 1.3])
with top_l:
    st.markdown(
        """
        <div class="topbar">
          <div class="brand">SAMSUNG ELECTRONICS · Process Monitor</div>
          <div class="subtitle">5-Stage sensor-based QC · Latest product view</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_c:
    st.markdown(
        f"""
        <div class="topbar">
          <div style="font-weight:900;font-size:18px;">Current (Latest) Run</div>
          <div class="subtitle">run={t} · id={cur["id"]} · window={w0}~{t} ({len(hist)} runs)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_r:
    status_color = banner_color
    ptxt = "-" if not np.isfinite(proba_ng) else f"{proba_ng*100:.1f}%"
    st.markdown(
        f"""
        <div class="topbar">
          <div class="rightmeta" style="font-size:18px;">RISK: <span style="color:{status_color}">{level_txt}</span></div>
          <div class="subtitle rightmeta">NG Prob={ptxt} · Label={int(cur["label"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

p_sub = "-" if not np.isfinite(proba_ng) else f"{proba_ng*100:.1f}%"
st.markdown(
    f"""
    <div class="banner {banner_class}">
      <div class="big">제품 불량 예측: {level_txt} · 불량 확률 {p_sub}</div>
      <div class="muted small">주의 기준 ≥ {MODEL_WARN_THRESHOLD*100:.0f}% · 위험 기준 ≥ {MODEL_RISK_THRESHOLD*100:.0f}%</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# KPI row (요청한 순서)
# ----------------------------
k1, k2, k3, k4 = st.columns(4)
kpi(k1, "총 생산량", float(total_prod), "ea", "{:.0f}")
kpi(k2, "불량 제품 수", float(defect_count), "ea", "{:.0f}")
kpi(k3, "불량 비율", defect_rate * 100 if np.isfinite(defect_rate) else np.nan, "%", "{:.2f}")
kpi(k4, "설비 이상 감지 수", float(equip_anom_count), "runs", "{:.0f}")

st.markdown("")

# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([2.55, 1.45], gap="medium")

with left:
    st.markdown('<div class="panel"><div class="pt">Stage Snapshot <span class="pill">온도/습도 + 편차(2)</span></div></div>', unsafe_allow_html=True)

    cols = st.columns(len(STAGES), gap="small")

    for i, s in enumerate(STAGES):
        tf = fcol(s, "temp")
        hf = fcol(s, "humidity")
        temp = pd.to_numeric(cur.get(tf), errors="coerce") if tf in df.columns else np.nan
        hum = pd.to_numeric(cur.get(hf), errors="coerce") if hf in df.columns else np.nan

        dev_ms = pick_stage_dev_metrics(s, 2)
        dev_vals = []
        for m in dev_ms:
            f = fcol(s, m)
            v = pd.to_numeric(cur.get(f), errors="coerce") if f in df.columns else np.nan
            dev_vals.append((m, v, f))

        stage_anom = stage_is_anomaly(s)
        dot_color = BAD if stage_anom else GOOD

        with cols[i]:
            parts = []
            parts.append('<div class="stagecard">')

            parts.append('<div class="stagehdr">')
            parts.append(f'<div class="stagetitle">Stage {s}</div>')
            parts.append(f'<div><span class="muted small">설비</span><span class="statusdot" style="background:{dot_color}"></span></div>')
            parts.append("</div>")

            parts.append('<div class="row2">')
            parts.append(f'<div class="m"><div class="ml">Temp (°C)</div><div class="mv">{fmtv(temp,2)}</div></div>')
            parts.append(f'<div class="m"><div class="ml">Humidity (%)</div><div class="mv">{fmtv(hum,2)}</div></div>')
            parts.append("</div>")

            for (m, v, f) in dev_vals:
                over = is_over_limit(v, f)
                color = BAD if over else TEXT
                parts.append('<div class="row2" style="margin-top:8px;">')
                parts.append(
                    f'<div class="m">'
                    f'<div class="ml">{m}</div>'
                    f'<div class="mv" style="color:{color};">{fmtv(v,2)}</div>'
                    f'<div class="muted small">limit: μ±{CTRL_K:.1f}σ</div>'
                    f"</div>"
                )
                parts.append("</div>")

            parts.append("</div>")
            st.markdown("".join(parts), unsafe_allow_html=True)

    st.markdown("")

    # ----------------------------
    # Recent Trend (auto-picked by max limit exceed)
    # ----------------------------
    st.markdown('<div class="panel"><div class="pt">Recent Trend <span class="pill">가장 위험한 한계선 초과 기준 자동 선택</span></div></div>', unsafe_allow_html=True)

    plot = hist[["run", "label", "id"]].copy()
    plot[trend_feature] = to_num(hist[trend_feature]) if trend_feature in hist.columns else np.nan
    plot = plot.dropna(subset=[trend_feature])

    mu, sd, lcl, ucl = CTRL_LIMITS.get(trend_feature, (np.nan, np.nan, np.nan, np.nan))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot["run"], y=plot[trend_feature], mode="lines+markers", name=trend_feature))

    bad_pts = plot[plot["label"] == 1]
    if len(bad_pts):
        fig.add_trace(
            go.Scatter(
                x=bad_pts["run"],
                y=bad_pts[trend_feature],
                mode="markers",
                name="NG(label=1)",
                marker=dict(size=8, symbol="x"),
            )
        )

    if t in set(plot["run"]):
        ycur = float(plot.loc[plot["run"] == t, trend_feature].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=[t],
                y=[ycur],
                mode="markers+text",
                name="current",
                text=["NOW"],
                textposition="top center",
                marker=dict(size=12, symbol="diamond"),
            )
        )

    if np.isfinite(lcl) and np.isfinite(ucl):
        fig.add_hline(y=ucl, line_dash="dot", annotation_text="UCL")
        fig.add_hline(y=lcl, line_dash="dot", annotation_text="LCL")

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        xaxis=dict(title="run (replay order)", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title=trend_feature, gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    # ----------------------------
    # Right (1): SHAP 영향도 Top5 (가로 막대)
    # ----------------------------
    st.markdown('<div class="panel"><div class="pt">제품 불량 영향도 <span class="pill">Top 5</span></div></div>', unsafe_allow_html=True)

    if shap_top5 is None or shap_top5.empty:
        st.markdown('<div class="muted small">SHAP 결과 없음 (모델/피처 확인 필요)</div>', unsafe_allow_html=True)
    else:
        # 보기 좋게 feature명을 짧게
        disp = shap_top5.copy()
        disp["feature"] = disp["feature"].astype(str)

        # Plotly horizontal bar (abs 기준 정렬, 방향은 shap 부호로)
        disp = disp.sort_values("abs_shap", ascending=True)

        fig_imp = go.Figure()
        fig_imp.add_trace(
            go.Bar(
                x=disp["shap"],
                y=disp["feature"],
                orientation="h",
                name="SHAP",
            )
        )
        fig_imp.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT),
            xaxis=dict(title="impact (SHAP)"),
            yaxis=dict(title="feature"),
            showlegend=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("")

    # ----------------------------
    # Right (2): 8 mini charts (Stage 1~5, y=metric)
    # ----------------------------
    st.markdown('<div class="panel"><div class="pt">현재 제품 · Stage Profile <span class="pill">8 metrics</span></div></div>', unsafe_allow_html=True)

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
    MINI_METRICS = [m for m in MINI_METRICS if m in METRICS]  # 데이터에 없는 건 제외

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

    # 4 columns x 2 rows (최대 8개)
    # metric이 8개 미만이면 가능한 만큼만 표시
    grid_cols = st.columns(4, gap="small")
    figs = []

    for m in MINI_METRICS[:8]:
        xs, ys = stage_series_for_metric(m)

        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=m))
        fig_m.update_layout(
            height=170,
            margin=dict(l=6, r=6, t=22, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, size=11),
            xaxis=dict(title="stage", tickmode="array", tickvals=xs, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title=m, gridcolor="rgba(255,255,255,0.06)"),
            showlegend=False,
            title=dict(text=m, x=0.02, y=0.98, xanchor="left", yanchor="top"),
        )
        figs.append(fig_m)

    # 4x2 placement
    for idx, fig_m in enumerate(figs):
        col = grid_cols[idx % 4]
        with col:
            st.plotly_chart(fig_m, use_container_width=True)