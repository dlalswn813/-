
import time
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================================================
# Operational Process Dashboard (Replay, Samsung-style)
# Upgrades for "real ops tool" feel:
# - Persistent event log (SQLite) + export
# - Alarm lifecycle: UNACK -> ACKED -> RESOLVED -> CLEARED
# - Escalation rule: consecutive OOC >= N triggers ESCALATE event
# - Theme switch + compact stage cards
# Note: Still replay (CSV); "real-time" needs streaming/DB integration.
# =========================================================

st.set_page_config(page_title="Process Monitor (Replay)", layout="wide")

DATA_PATH = "mice_final_data_with_id.csv"
DB_PATH = "ops_log.db"

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

df = load_csv(DATA_PATH)

if "id" not in df.columns or "label" not in df.columns:
    st.error("CSV에 'id', 'label' 컬럼이 필요합니다.")
    st.stop()

# deterministic replay order
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
# SQLite: persistent log + run state
# ----------------------------
def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_iso TEXT NOT NULL,
            ts_hms TEXT NOT NULL,
            run INTEGER NOT NULL,
            sample_id TEXT NOT NULL,
            etype TEXT NOT NULL,
            msg TEXT NOT NULL,
            user TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_state (
            run INTEGER PRIMARY KEY,
            sample_id TEXT NOT NULL,
            acked INTEGER DEFAULT 0,
            resolved INTEGER DEFAULT 0,
            escalated INTEGER DEFAULT 0,
            updated_ts_iso TEXT
        )
    """)
    conn.commit()
    return conn

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def now_hms():
    return time.strftime("%H:%M:%S")

def add_event(conn, run, sample_id, etype, msg, user=None):
    conn.execute(
        "INSERT INTO events(ts_iso, ts_hms, run, sample_id, etype, msg, user) VALUES (?,?,?,?,?,?,?)",
        (now_iso(), now_hms(), int(run), str(sample_id), str(etype), str(msg), None if user is None else str(user)),
    )
    conn.commit()

def get_state(conn, run, sample_id):
    cur = conn.execute("SELECT acked, resolved, escalated FROM run_state WHERE run=?", (int(run),))
    row = cur.fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO run_state(run, sample_id, acked, resolved, escalated, updated_ts_iso) VALUES (?,?,?,?,?,?)",
            (int(run), str(sample_id), 0, 0, 0, now_iso()),
        )
        conn.commit()
        return {"acked": 0, "resolved": 0, "escalated": 0}
    return {"acked": int(row[0]), "resolved": int(row[1]), "escalated": int(row[2])}

def set_state(conn, run, sample_id, **kwargs):
    state = get_state(conn, run, sample_id)
    acked = int(kwargs.get("acked", state["acked"]))
    resolved = int(kwargs.get("resolved", state["resolved"]))
    escalated = int(kwargs.get("escalated", state["escalated"]))
    conn.execute(
        "UPDATE run_state SET sample_id=?, acked=?, resolved=?, escalated=?, updated_ts_iso=? WHERE run=?",
        (str(sample_id), acked, resolved, escalated, now_iso(), int(run)),
    )
    conn.commit()

def fetch_events(conn, limit=200):
    cur = conn.execute(
        "SELECT ts_hms, run, sample_id, etype, msg, COALESCE(user,'') FROM events ORDER BY id DESC LIMIT ?",
        (int(limit),),
    )
    rows = cur.fetchall()
    rows.reverse()
    return pd.DataFrame(rows, columns=["time", "run", "id", "type", "msg", "user"])

conn = db_conn()

# ----------------------------
# Sidebar: Theme + Replay + Spec + Trend selection
# ----------------------------
st.sidebar.markdown("### Process Monitor")

theme = st.sidebar.selectbox("Theme", ["Black (default)", "Dark Navy", "Dark Gray"], index=0)

# Samsung-like brand colors
SAMSUNG_BLUE_2 = "#1E40FF"
GOOD = "#22C55E"
WARN = "#F59E0B"
BAD = "#EF4444"
TEXT = "#E9EDF6"
MUTED = "#AAB6D3"

if theme == "Black (default)":
    BG = "#05070D"; PANEL = "#0F1628"; STROKE = "#1E2942"
elif theme == "Dark Navy":
    BG = "#0B0F1A"; PANEL = "#111829"; STROKE = "#25304A"
else:
    BG = "#0C0D10"; PANEL = "#14161C"; STROKE = "#2B2E37"

st.sidebar.markdown("---")
operator = st.sidebar.text_input("Operator", value="operator1")

auto = st.sidebar.toggle("Auto Play", value=False)
interval = st.sidebar.slider("Update interval (sec)", 0.2, 3.0, 0.8, 0.1)
speed = st.sidebar.slider("Step size (rows)", 1, 20, 1, 1)
window_n = st.sidebar.slider("Recent window (runs)", 20, 500, 120, 10)

compact = st.sidebar.toggle("Compact stage cards", value=True)

escalate_n = st.sidebar.slider("Escalate if Consecutive OOC ≥", 2, 20, 5, 1)

cA, cB = st.sidebar.columns(2)
if cA.button("⏮ Reset"):
    st.session_state["t"] = 0
if cB.button("⏭ Next"):
    st.session_state["t"] = min(st.session_state.get("t", 0) + speed, len(df) - 1)

use_slider = st.sidebar.toggle("Manual seek", value=False)
if "t" not in st.session_state:
    st.session_state["t"] = 0
if use_slider:
    st.session_state["t"] = st.sidebar.slider("Run pointer", 0, len(df) - 1, int(st.session_state["t"]), 1)

t = int(st.session_state["t"])
cur = df.iloc[t]

st.sidebar.markdown("---")
st.sidebar.markdown("### Spec limits (|deviation|)")
default_specs = {
    "flow_deviation": 10.0,
    "density_deviation": 10.0,
    "viscosity_deviation": 10.0,
    "o2_deviation": 5.0,
    "n_deviation": 5.0,
    "co2_deviation": 10.0,
}
spec = {}
for m in DEV_METRICS:
    spec[m] = st.sidebar.number_input(
        f"{m}",
        min_value=0.1,
        max_value=100.0,
        value=float(default_specs.get(m, 10.0)),
        step=0.5,
    )

st.sidebar.markdown("---")
trend_stage = st.sidebar.selectbox("Trend Stage", STAGES, index=0)
trend_metric = st.sidebar.selectbox("Trend Metric", METRICS, index=(METRICS.index("co2_deviation") if "co2_deviation" in METRICS else 0))
trend_feature = fcol(int(trend_stage), trend_metric)

# Export log
st.sidebar.markdown("---")
if st.sidebar.button("Export events CSV"):
    ev = fetch_events(conn, limit=5000)
    ev.to_csv("events_export.csv", index=False)
    st.sidebar.success("Saved: events_export.csv (same folder)")

# ----------------------------
# CSS
# ----------------------------
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
      [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(8,11,20,1), var(--bg));
        border-right: 1px solid var(--stroke);
      }}
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
      .banner.alarm {{
        border-color: rgba(239,68,68,0.45);
        background: rgba(239,68,68,0.10);
      }}
      .banner.acked {{
        border-color: rgba(245,158,11,0.45);
        background: rgba(245,158,11,0.10);
      }}
      .banner.resolved {{
        border-color: rgba(34,197,94,0.40);
        background: rgba(34,197,94,0.10);
      }}
      .banner.normal {{
        border-color: rgba(34,197,94,0.25);
        background: rgba(34,197,94,0.06);
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
        min-height: 175px;
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
# OOC + near-spec + rolling
# ----------------------------
def is_ooc(value, metric):
    if metric not in spec:
        return False
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return abs(float(value)) > float(spec[metric])

def near_spec_ratio(value, metric):
    if metric not in spec:
        return np.nan
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    lim = float(spec[metric])
    if lim <= 0:
        return np.nan
    return abs(float(value)) / lim

cur_ooc = {}
cur_ooc_count = 0
near_max = 0.0
issues = []  # (ooc, ratio, stage, metric, value)

for s in STAGES:
    cur_ooc[s] = []
    for m in DEV_METRICS:
        f = fcol(s, m)
        v = to_num(pd.Series([cur.get(f)])).iloc[0] if f in df.columns else np.nan
        ooc = is_ooc(v, m)
        r = near_spec_ratio(v, m)
        if np.isfinite(r):
            near_max = max(near_max, float(r))
            issues.append((ooc, float(r), s, m, float(v) if pd.notna(v) else np.nan))
        cur_ooc[s].append((m, float(v) if pd.notna(v) else np.nan, ooc, r))
        if ooc:
            cur_ooc_count += 1

issues.sort(key=lambda x: (x[0], x[1]), reverse=True)
top_issues = issues[:3]

w0 = max(0, t - window_n + 1)
hist = df.iloc[w0 : t + 1].copy()

ooc_counts = []
ooc_metric_hits = {m: 0 for m in DEV_METRICS}
for _, r in hist.iterrows():
    cnt = 0
    for s in STAGES:
        for m in DEV_METRICS:
            f = fcol(s, m)
            if f in df.columns:
                vv = to_num(pd.Series([r.get(f)])).iloc[0]
                if is_ooc(vv, m):
                    cnt += 1
                    ooc_metric_hits[m] += 1
    ooc_counts.append(cnt)
hist["ooc_count"] = ooc_counts

# consecutive OOC at tail
consecutive_ooc = 0
for i in range(len(hist) - 1, -1, -1):
    if hist.iloc[i]["ooc_count"] > 0:
        consecutive_ooc += 1
    else:
        break

ng_rate = float(hist["label"].mean()) if len(hist) else np.nan
avg_ooc = float(hist["ooc_count"].mean()) if len(hist) else np.nan
is_ng = int(cur["label"]) == 1
alarm = is_ng or (cur_ooc_count > 0)

top_metric = max(ooc_metric_hits.items(), key=lambda kv: kv[1])[0] if len(ooc_metric_hits) else None

reasons = []
for (ooc, r, s, m, v) in issues:
    if ooc:
        reasons.append(f"Stage{s} {m}: {v:+.2f} (spec ±{spec[m]:.1f})")
reasons = reasons[:8]

ACTION_MAP = {
    "flow_deviation": ["MFC/유량 제어기 상태 확인", "라인 막힘/누설 여부 점검", "setpoint/recipe 유량값 재확인"],
    "density_deviation": ["약품 농도/혼합 비율 점검", "공급 탱크/라인 기포·침전 확인", "센서 캘리브레이션 상태 확인"],
    "viscosity_deviation": ["약품 온도/점도 보정 조건 확인", "필터/라인 막힘 점검", "혼합/순환 조건 점검"],
    "o2_deviation": ["가스 공급/퍼지 조건 확인", "챔버 누설/배기 상태 점검", "센서 드리프트(보정) 확인"],
    "n_deviation": ["N2 공급/밸브 상태 확인", "퍼지/시퀀스 조건 확인", "센서 보정/교체 필요 여부 점검"],
    "co2_deviation": ["가스 공급/혼합 비율 확인", "라인 누설/백프레셔 확인", "챔버 배기/압력 제어(APC) 상태 확인"],
}
def build_actions(top_issues):
    actions = []
    seen = set()
    for (ooc, r, s, m, v) in top_issues:
        if m in ACTION_MAP and m not in seen:
            seen.add(m)
            actions.extend([f"[{m}] {a}" for a in ACTION_MAP[m]])
    return actions[:10]
actions = build_actions(top_issues)

# ----------------------------
# Persistent lifecycle state
# ----------------------------
state = get_state(conn, t, cur["id"])
acked = bool(state["acked"])
resolved = bool(state["resolved"])
escalated = bool(state["escalated"])

# Escalation (once per run): consecutive OOC >= threshold AND alarm
if alarm and (consecutive_ooc >= int(escalate_n)) and (not escalated):
    set_state(conn, t, cur["id"], escalated=1)
    add_event(conn, t, cur["id"], "ESCALATE", f"Consecutive OOC={consecutive_ooc} >= {escalate_n}", operator)
    escalated = True

# State transitions logging (persistent): ALARM start/clear
if "last_state_sig" not in st.session_state:
    st.session_state["last_state_sig"] = None
sig = (t, int(alarm), int(is_ng), int(cur_ooc_count))
if st.session_state["last_state_sig"] != sig:
    add_event(conn, t, cur["id"], "STATE", f"{'ALARM' if alarm else 'NORMAL'} (NG={int(is_ng)}, OOC={cur_ooc_count})", operator)
    st.session_state["last_state_sig"] = sig

# ----------------------------
# Determine banner class / status text
# ----------------------------
if alarm and resolved:
    banner_class = "resolved"
    banner_title = "ALARM (RESOLVED) · 조치 완료(모니터링 지속)"
    banner_sub = "조치 완료로 표시됨. 상태가 정상으로 복귀하는지 확인하세요."
elif alarm and acked:
    banner_class = "acked"
    banner_title = "ALARM (ACKED) · 확인 처리됨"
    banner_sub = "확인(ACK) 처리됨. 점검/조치 진행 후 RESOLVED로 전환하세요."
elif alarm:
    banner_class = "alarm"
    banner_title = "ALARM 발생 · 즉시 확인 필요"
    banner_sub = f"NG={int(is_ng)} · OOC={cur_ooc_count} · TOP: {('Stage'+str(top_issues[0][2])+' '+top_issues[0][3]) if top_issues else '-'}"
else:
    banner_class = "normal"
    banner_title = "NORMAL · 운영 안정"
    banner_sub = "최근 window 기준 OOC/NG가 낮고 spec 근접도도 안정적입니다."

# ----------------------------
# Top bar + banner
# ----------------------------
top_l, top_c, top_r = st.columns([1.7, 2.0, 1.3])
with top_l:
    st.markdown(
        '''
        <div class="topbar">
          <div class="brand">SAMSUNG ELECTRONICS · Process Monitor</div>
          <div class="subtitle">5-Stage sensor-based QC · Replay streaming demo</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
with top_c:
    st.markdown(
        f'''
        <div class="topbar">
          <div style="font-weight:900;font-size:18px;">Current Run</div>
          <div class="subtitle">run={t} · id={cur["id"]} · window={w0}~{t} ({len(hist)} runs)</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
with top_r:
    status_txt = "ALARM" if alarm else "NORMAL"
    status_color = BAD if alarm else GOOD
    ack_txt = "ACKED" if acked else "UNACK"
    res_txt = "RESOLVED" if resolved else "UNRESOLVED"
    esc_txt = "ESC" if escalated else "-"
    st.markdown(
        f'''
        <div class="topbar">
          <div class="rightmeta" style="font-size:18px;">STATUS: <span style="color:{status_color}">{status_txt}</span></div>
          <div class="subtitle rightmeta">ACK={ack_txt} · {res_txt} · ESC={esc_txt} · NG={int(is_ng)} · OOC={cur_ooc_count} · NearSpec={near_max*100:.0f}%</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

st.markdown(
    f'''
    <div class="banner {banner_class}">
      <div class="big">{banner_title}</div>
      <div class="muted small">{banner_sub}</div>
    </div>
    ''',
    unsafe_allow_html=True,
)

# ----------------------------
# KPI row
# ----------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)
kpi(k1, "NG rate (recent)", ng_rate * 100 if np.isfinite(ng_rate) else np.nan, "%", "{:.2f}")
kpi(k2, "Avg OOC (recent)", avg_ooc if np.isfinite(avg_ooc) else np.nan, "count", "{:.2f}")
kpi(k3, "Current OOC", float(cur_ooc_count), "count", "{:.0f}")
kpi(k4, "Consecutive OOC", float(consecutive_ooc), "runs", "{:.0f}")
kpi(k5, "Recurring metric", 1.0 if top_metric else 0.0, "", "{:.0f}")
kpi(k6, "Near-spec max", near_max * 100 if np.isfinite(near_max) else np.nan, "%", "{:.0f}")
st.markdown(f"<div class='muted small'>Recurring metric (window): <b>{top_metric if top_metric else '-'}</b></div>", unsafe_allow_html=True)

st.markdown("")

# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([2.55, 1.45], gap="medium")

with left:
    st.markdown('<div class="panel"><div class="pt">Stage Status <span class="pill">temp/humidity + deviation OOC</span></div></div>', unsafe_allow_html=True)

    cols = st.columns(len(STAGES), gap="small")
    key_dev = [m for m in ["flow_deviation","density_deviation","viscosity_deviation","o2_deviation","n_deviation","co2_deviation"] if m in DEV_METRICS]

    for i, s in enumerate(STAGES):
        tf = fcol(s, "temp")
        hf = fcol(s, "humidity")
        temp = to_num(pd.Series([cur.get(tf)])).iloc[0] if tf in df.columns else np.nan
        hum = to_num(pd.Series([cur.get(hf)])).iloc[0] if hf in df.columns else np.nan

        stage_ooc = any(ooc for (_, _, ooc, _) in cur_ooc[s])
        dot = BAD if stage_ooc else GOOD

        items = {m: next((v for (mm, v, o, r) in cur_ooc[s] if mm == m), np.nan) for m in key_dev}
        flags = {m: next((o for (mm, v, o, r) in cur_ooc[s] if mm == m), False) for m in key_dev}
        ratios = {m: next((r for (mm, v, o, r) in cur_ooc[s] if mm == m), np.nan) for m in key_dev}

        show_list = list(items.keys())
        if compact and len(show_list) > 2:
            ranked = []
            for m in show_list:
                rr = ratios[m]
                oo = flags[m]
                score = (2.0 if oo else 0.0) + (float(rr) if np.isfinite(rr) else 0.0)
                ranked.append((score, m))
            ranked.sort(reverse=True)
            show_list = [m for (_, m) in ranked[:2]]

        with cols[i]:
            parts = []
            parts.append('<div class="stagecard">')
            parts.append('<div class="stagehdr">')
            parts.append(f'<div class="stagetitle">Stage {s}</div>')
            parts.append(f'<div><span class="muted small">OOC</span><span class="statusdot" style="background:{dot}"></span></div>')
            parts.append('</div>')

            parts.append('<div class="row2">')
            parts.append(f'<div class="m"><div class="ml">Temp (°C)</div><div class="mv">{fmtv(temp,2)}</div></div>')
            parts.append(f'<div class="m"><div class="ml">Humidity (%)</div><div class="mv">{fmtv(hum,2)}</div></div>')
            parts.append('</div>')

            for m in show_list:
                v = items[m]
                o = flags[m]
                r = ratios[m]
                if o:
                    color = BAD
                elif np.isfinite(r) and r >= 0.8:
                    color = WARN
                else:
                    color = TEXT
                parts.append('<div class="row2" style="margin-top:8px;">')
                parts.append(
                    f'<div class="m"><div class="ml">{m}</div>'
                    f'<div class="mv" style="color:{color};">{fmtv(v,2)}</div>'
                    f'<div class="muted small">spec ±{spec[m]:.1f} · {("near "+str(int(r*100))+"%") if np.isfinite(r) else ""}</div></div>'
                )
                parts.append('</div>')

            if compact and len(key_dev) > 2:
                parts.append('<div class="muted small" style="margin-top:6px;">(compact: top 2 deviations)</div>')

            parts.append('</div>')
            st.markdown("".join(parts), unsafe_allow_html=True)

    st.markdown("")

    st.markdown('<div class="panel"><div class="pt">Recent Trend <span class="pill">selected stage/metric</span></div></div>', unsafe_allow_html=True)

    plot = hist[["run", "label", "id"]].copy()
    if trend_feature in hist.columns:
        plot[trend_feature] = to_num(hist[trend_feature])
        plot = plot.dropna(subset=[trend_feature])
    else:
        plot[trend_feature] = np.nan

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot["run"], y=plot[trend_feature], mode="lines+markers", name=trend_feature))

    bad_pts = plot[plot["label"] == 1]
    if len(bad_pts):
        fig.add_trace(go.Scatter(x=bad_pts["run"], y=bad_pts[trend_feature], mode="markers",
                                 name="NG(label=1)", marker=dict(size=9, symbol="x")))

    if t in set(plot["run"]):
        ycur = float(plot.loc[plot["run"] == t, trend_feature].iloc[0])
        fig.add_trace(go.Scatter(x=[t], y=[ycur], mode="markers+text", name="current",
                                 text=["NOW"], textposition="top center", marker=dict(size=12, symbol="diamond")))

    if trend_metric in spec:
        lim = float(spec[trend_metric])
        fig.add_hline(y=lim, line_dash="dot", annotation_text="spec +")
        fig.add_hline(y=-lim, line_dash="dot", annotation_text="spec -")

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
    st.markdown('<div class="panel"><div class="pt">Operator Panel <span class="pill">Lifecycle</span></div></div>', unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)

    # ACK: mark acknowledged
    if b1.button("ACK", use_container_width=True, disabled=(not alarm) or acked):
        set_state(conn, t, cur["id"], acked=1)
        add_event(conn, t, cur["id"], "ACK", "ACK confirmed", operator)
        st.rerun()

    # RESOLVED: after actions done (still may be alarm)
    if b2.button("RESOLVED", use_container_width=True, disabled=(not alarm) or resolved):
        set_state(conn, t, cur["id"], resolved=1, acked=1)
        add_event(conn, t, cur["id"], "RESOLVED", "Marked as resolved by operator", operator)
        st.rerun()

    # CLEAR: only allow when current state is normal
    if b3.button("CLEAR", use_container_width=True, disabled=alarm):
        set_state(conn, t, cur["id"], acked=0, resolved=0, escalated=0)
        add_event(conn, t, cur["id"], "CLEAR", "Alarm cleared (state reset)", operator)
        st.rerun()

    st.markdown("---")
    st.markdown("**TOP 3 Issues (현재 Run)**")
    if top_issues:
        for (ooc, r, s, m, v) in top_issues:
            tag = "OOC" if ooc else "NEAR"
            pct = f"{int(r*100)}%" if np.isfinite(r) else "-"
            st.markdown(f"- **Stage{s} {m}**: `{v:+.2f}` · {tag} · near {pct}")
    else:
        st.markdown("- (유효 deviation 데이터 없음)")

    st.markdown("---")
    st.markdown("**Action checklist (추천)**")
    if actions:
        for a in actions:
            st.markdown(f"- {a}")
    else:
        st.markdown("- (OOC/near-spec 기반 액션 템플릿 없음)")

    st.markdown("---")
    st.markdown("**Alarm**")
    if alarm:
        if resolved:
            st.success(f"ALARM (RESOLVED) · NG={int(is_ng)} · OOC={cur_ooc_count} · consecutive={consecutive_ooc}")
        elif acked:
            st.warning(f"ALARM (ACKED) · NG={int(is_ng)} · OOC={cur_ooc_count} · consecutive={consecutive_ooc}")
        else:
            st.error(f"ALARM · NG={int(is_ng)} · OOC={cur_ooc_count} · consecutive={consecutive_ooc}")
        if escalated:
            st.error("ESCALATED: 연속 OOC 기준 초과")
        if reasons:
            st.markdown("**OOC reasons**")
            for r in reasons:
                st.markdown(f"- {r}")
        else:
            st.markdown("- NG만 발생, OOC 없음(또는 spec 미설정)")
    else:
        st.success("NORMAL · NG=0 · OOC=0")

    st.markdown("---")
    st.markdown("**Event Feed (SQLite)**")
    feed = fetch_events(conn, limit=200)
    st.dataframe(feed, use_container_width=True, height=280)

    st.markdown('<div class="muted small">※ 로그는 ops_log.db에 영속 저장됨. (서버 재시작/새로고침 후에도 유지)</div>', unsafe_allow_html=True)

# ----------------------------
# Autoplay
# ----------------------------
if auto and not use_slider:
    if t < len(df) - 1:
        time.sleep(float(interval))
        st.session_state["t"] = min(t + speed, len(df) - 1)
        st.rerun()
