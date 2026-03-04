import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import re
import plotly.graph_objects as go
from pathlib import Path

# ✅ AgGrid (설치: pip install streamlit-aggrid)
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

st.set_page_config(layout="wide", page_title="센서 알림")

# ✅ Altair 안정화(행 제한 해제 + 임베드 액션 제거)
alt.data_transformers.disable_max_rows()
alt.renderers.set_embed_options(actions=False)

# ============================
# CONFIG
# ============================
HERE = Path(__file__).resolve()
ROOT_DIR = HERE.parents[1] if HERE.parent.name == "pages" else HERE.parent

MODEL_PATH = str(ROOT_DIR / "catboost_final_model.cbm")
DATA_PATH  = str(ROOT_DIR / "mice_final_data_with_id.csv")

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

SENSOR_TYPE_OPTIONS = [
    "온도",
    "습도",
    "유량 편차",
    "밀도 편차",
    "점도 편차",
    "CO₂ 편차",
    "O₂ 편차",
    "N₂ 편차",
]
KO_TO_SUFFIX = {v: k for k, v in SENSOR_SUFFIX_TO_KO.items()}

_STAGE_RE = re.compile(r"^stage(\d+)_(.+)$", re.IGNORECASE)


def parse_stage(sensor_col: str) -> int | None:
    m = _STAGE_RE.match(sensor_col)
    return int(m.group(1)) if m else None


def parse_suffix(sensor_col: str) -> str | None:
    m = _STAGE_RE.match(sensor_col)
    return m.group(2) if m else None


def parse_process(sensor_col: str) -> str:
    n = parse_stage(sensor_col)
    return f"Stage {n}" if n is not None else "미분류"


def sensor_display_name(sensor_col: str) -> str:
    suf = parse_suffix(sensor_col)
    if suf and suf in SENSOR_SUFFIX_TO_KO:
        return SENSOR_SUFFIX_TO_KO[suf]
    return sensor_col


def stage_to_int(stage_label: str) -> int | None:
    m = re.match(r"^stage\s*(\d+)$", stage_label.strip(), re.IGNORECASE)
    return int(m.group(1)) if m else None


# ============================
# CSS (메인 공통)
# ============================
st.markdown(
    """
<style>
div.block-container{
  padding-top: 2.3rem;
  padding-bottom: 0.75rem;
  padding-left: 1.25rem;
  padding-right: 1.25rem;
}
h1, h2, h3 { margin-top: 0.2rem !important; margin-bottom: 0.55rem !important; }
hr { margin: 0.7rem 0 !important; }
[data-testid="stVerticalBlock"] { gap: 0.55rem; }
[data-testid="stMultiSelect"] > div { max-width: 420px; }

/* KPI */
.kpi-grid{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
  margin-top: 6px;
  margin-bottom: 2px;
}
.kpi-card{
  border: 1px solid rgba(49, 51, 63, 0.18);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.6);
  box-shadow: 0 1px 8px rgba(0,0,0,0.04);
}
.kpi-label{
  font-size: 0.85rem;
  color: rgba(49, 51, 63, 0.7);
  margin-bottom: 6px;
}
.kpi-value{
  font-size: 1.85rem;
  font-weight: 700;
  line-height: 1.1;
  color: rgba(49, 51, 63, 0.98);
}

/* 작은 정사각형 버튼 */
.square-btn button{
  width: 44px !important;
  height: 44px !important;
  padding: 0 !important;
  border-radius: 12px !important;
  font-size: 0.95rem !important;
}

/* 팝오버 버튼 폭 과대 방지 */
.compact-popover button{
  padding: 0.35rem 0.6rem !important;
  font-size: 0.9rem !important;
}

/* 알람 테이블 안내 문구 */
.section-head{
  display:flex;
  align-items: baseline;
  gap:0.1px;  
  margin: 0 0 1px 0; 
}
.section-head .hint{
  color: rgba(49, 51, 63, 0.55);
  font-size: 0.92rem;
  line-height: 1.2;
}

/* 조건 요약 칩 */
.chip-row{ display:flex; flex-wrap:wrap; gap:8px; margin: 2px 0 8px 0; }
.chip{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(49, 51, 63, 0.18);
  background: rgba(255,255,255,0.75);
  font-size: 0.82rem;
  color: rgba(49, 51, 63, 0.88);
}
.chip b{ font-weight: 650; }
</style>
""",
    unsafe_allow_html=True,
)


def kpi_cards(alarm_cnt: int, sensor_cnt: int, product_cnt: int):
    st.markdown(
        f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">대상 알람(건수)</div>
    <div class="kpi-value">{alarm_cnt:,}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">총 계측 포인트 수</div>
    <div class="kpi-value">{sensor_cnt:,}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">총 제품 수</div>
    <div class="kpi-value">{product_cnt:,}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_condition_chips(selected_processes: list[str], selected_types: list[str]):
    stages_txt = ", ".join(selected_processes) if selected_processes else "전체"
    types_txt = ", ".join(selected_types) if selected_types else "전체"
    st.markdown(
        f"""
<div class="chip-row">
  <span class="chip"><b>공정 단계</b> {stages_txt}</span>
  <span class="chip"><b>측정 항목</b> {types_txt}</span>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================
# Utils
# ============================
def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


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


def is_id_col(c: str) -> bool:
    c2 = c.lower()
    return c2 == "id" or c2.endswith("_id") or c2 == "product_id"


def make_detail_chart(df_plot: pd.DataFrame, limits: dict, y_title: str, selected_order: int | None):
    # ✅ dtype 안정화 (렌더러 조용히 죽는 케이스 방지)
    dfp = df_plot.copy()
    dfp["order"] = pd.to_numeric(dfp["order"], errors="coerce")
    dfp["value"] = pd.to_numeric(dfp["value"], errors="coerce")
    dfp["id"] = dfp["id"].astype(str)

    base = alt.Chart(dfp).encode(
        x=alt.X("order:Q", axis=None),
        y=alt.Y("value:Q", title=y_title),
        tooltip=[
            alt.Tooltip("id:N", title="제품 ID"),
            alt.Tooltip("value:Q", title=y_title, format=".4f"),
            alt.Tooltip("flag:N", title="상태"),
        ],
    )

    line = base.mark_line()

    pts_ooc = (
        base.transform_filter("datum.flag == 'OOC'")
        .mark_circle(size=45, color="red")
    )

    layers = [line, pts_ooc]

    # ✅ datum rule 방식(레이어 안정)
    if np.isfinite(limits.get("mean", np.nan)):
        layers.append(alt.Chart(dfp).mark_rule().encode(y=alt.datum(float(limits["mean"]))))

    if np.isfinite(limits.get("ucl", np.nan)):
        layers.append(alt.Chart(dfp).mark_rule().encode(y=alt.datum(float(limits["ucl"]))))

    if np.isfinite(limits.get("lcl", np.nan)):
        layers.append(alt.Chart(dfp).mark_rule().encode(y=alt.datum(float(limits["lcl"]))))

    if selected_order is not None:
        layers.append(
            alt.Chart(dfp)
            .mark_rule(strokeDash=[6, 4], opacity=0.85)
            .encode(x=alt.datum(float(selected_order)))
        )
        layers.append(
            base.transform_filter(f"datum.order == {int(selected_order)}")
            .mark_circle(size=160, color="#ff7f0e")
        )

    return alt.layer(*layers).properties(height=420).interactive()


# ============================
# Load data
# ============================
@st.cache_data(show_spinner=False)
def load_csv_by_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


df = load_csv_by_path(DATA_PATH)

cols = df.columns.tolist()
id_col = next((c for c in cols if is_id_col(c)), None)
if id_col is None:
    st.error("ID를 찾을 수 없습니다.")
    st.stop()

sensor_cols = [c for c in cols if c != id_col and "label" not in c.lower() and "fail" not in c.lower()]
for c in sensor_cols:
    df[c] = safe_num(df[c])

df = df.sort_values(id_col).reset_index(drop=True)
df["order"] = np.arange(len(df))
df = df.rename(columns={id_col: "id"})

# ============================
# Session state
# ============================
if "page" not in st.session_state:
    st.session_state.page = "main"
if "selected_event" not in st.session_state:
    st.session_state.selected_event = None
if "selected_sensor" not in st.session_state:
    st.session_state.selected_sensor = sensor_cols[0] if sensor_cols else None

# main filters
if "selected_processes" not in st.session_state:
    st.session_state.selected_processes = []
if "selected_sensor_types" not in st.session_state:
    st.session_state.selected_sensor_types = []

# detail filters
if "detail_stage_one" not in st.session_state:
    st.session_state.detail_stage_one = None
if "detail_type_one" not in st.session_state:
    st.session_state.detail_type_one = None
if "detail_before_n" not in st.session_state:
    st.session_state.detail_before_n = 200
if "detail_after_n" not in st.session_state:
    st.session_state.detail_after_n = 200

# detail change toast tracking
if "detail_prev_stage" not in st.session_state:
    st.session_state.detail_prev_stage = None
if "detail_prev_type" not in st.session_state:
    st.session_state.detail_prev_type = None

# 루프 방지: 메인 선택 상태 기억/해제용
if "selected_alarm_event_id" not in st.session_state:
    st.session_state.selected_alarm_event_id = None


# ============================
# Limits + Alarm table
# ============================
@st.cache_data(show_spinner=False)
def build_limits(df_in: pd.DataFrame, sensors: list[str]) -> dict:
    return {s: calc_imr_limits(df_in[s].to_numpy(dtype=float)) for s in sensors}


limits_by_sensor = build_limits(df, sensor_cols)


def resolve_sensor_columns(selected_processes: list[str], selected_sensor_types_ko: list[str]) -> list[str]:
    stage_set = None
    if selected_processes:
        stage_nums = [stage_to_int(s) for s in selected_processes]
        stage_nums = [n for n in stage_nums if n is not None]
        stage_set = set(stage_nums) if stage_nums else None

    suffix_set = None
    if selected_sensor_types_ko:
        suffixes = [KO_TO_SUFFIX.get(k) for k in selected_sensor_types_ko]
        suffixes = [s for s in suffixes if s]
        suffix_set = set(suffixes) if suffixes else None

    out = []
    for col in sensor_cols:
        stg = parse_stage(col)
        suf = parse_suffix(col)
        if stg is None or suf is None:
            continue
        if stage_set is not None and stg not in stage_set:
            continue
        if suffix_set is not None and suf not in suffix_set:
            continue
        out.append(col)
    return out


def build_alarm_table(df_in: pd.DataFrame, sensors_filtered: list[str], limits_map: dict):
    rows = []
    for s in sensors_filtered:
        lim = limits_map[s]
        ucl, lcl = lim["ucl"], lim["lcl"]
        v = df_in[s].to_numpy(dtype=float)

        if not (np.isfinite(ucl) and np.isfinite(lcl)):
            continue

        ooc = np.isfinite(v) & ((v > ucl) | (v < lcl))
        idxs = np.where(ooc)[0]

        for i in idxs:
            # 이탈 방향(+/-) + 이탈폭(해당 한계선 기준)
            if np.isfinite(v[i]) and v[i] > ucl:
                direction = "+"
                dist = float(abs(v[i] - ucl))
            else:
                direction = "-"
                dist = float(abs(v[i] - lcl))

            rows.append(
                {
                    "event_id": f"{s}::{df_in.loc[i,'id']}",
                    "공정 단계": parse_process(s),
                    "측정 항목": sensor_display_name(s),
                    "sensor_raw": s,
                    "제품 ID": str(df_in.loc[i, "id"]),
                    "order": int(df_in.loc[i, "order"]),
                    "이탈 방향": direction,
                    "한계선 이탈폭": f"{direction}{dist:.3f}",
                    "측정값": float(v[i]),
                }
            )

    alarms = pd.DataFrame(rows)
    if alarms.empty:
        return alarms
    return alarms.sort_values(["order"], ascending=[False]).reset_index(drop=True)


# ============================
# UI
# ============================
all_processes = sorted({parse_process(s) for s in sensor_cols})

if st.session_state.page == "main":
    st.markdown("## 설비 이상 알람 현황")

    kpi_ph = st.empty()
    st.divider()

    left, right = st.columns([4.8, 1.6], gap="large")

    with right:
        st.markdown('<div class="compact-popover">', unsafe_allow_html=True)
        with st.popover("조건", use_container_width=True):
            st.caption("미선택 시 전체 항목이 조회됩니다.")
            c1, c2 = st.columns(2, gap="small")

            with c1:
                st.caption("공정 단계")
                stage_checked = []
                for s in all_processes:
                    checked = st.checkbox(
                        s,
                        value=(s in st.session_state.selected_processes),
                        key=f"main_stage_cb_{s}",
                    )
                    if checked:
                        stage_checked.append(s)

            with c2:
                st.caption("측정 항목")
                type_checked = []
                for t in SENSOR_TYPE_OPTIONS:
                    checked = st.checkbox(
                        t,
                        value=(t in st.session_state.selected_sensor_types),
                        key=f"main_type_cb_{t}",
                    )
                    if checked:
                        type_checked.append(t)

            st.session_state.selected_processes = stage_checked
            st.session_state.selected_sensor_types = type_checked
        st.markdown("</div>", unsafe_allow_html=True)

    sensors_filtered = resolve_sensor_columns(st.session_state.selected_processes, st.session_state.selected_sensor_types)
    alarms = build_alarm_table(df, sensors_filtered, limits_by_sensor)

    with kpi_ph:
        kpi_cards(alarm_cnt=len(alarms), sensor_cnt=len(sensor_cols), product_cnt=len(df))

    render_condition_chips(st.session_state.selected_processes, st.session_state.selected_sensor_types)

    st.divider()

    with left:
        st.markdown(
            """
        <div class="section-head">
        <h3 style="margin:0;">알람 내역</h3>
        <span class="hint">행을 클릭하면 해당 제품의 센서 관리도 상세로 이동합니다.</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if alarms.empty:
            st.info("현재 조회 조건에서 발생한 알람이 없습니다.")
        else:
            TABLE_HEIGHT = 560

            grid_df = alarms[
                ["event_id", "제품 ID", "공정 단계", "측정 항목", "sensor_raw", "order", "한계선 이탈폭", "측정값"]
            ].copy()

            gb = GridOptionsBuilder.from_dataframe(grid_df)
            gb.configure_default_column(
                flex=1,
                minWidth=130,
                resizable=True,
                sortable=True,
                filter=True,
            )
            gb.configure_selection(selection_mode="single", use_checkbox=False)
            gb.configure_column("event_id", hide=True)
            gb.configure_column("sensor_raw", hide=True)
            gb.configure_column("order", hide=True)
            gb.configure_column("측정값", type=["numericColumn"], valueFormatter="x.toFixed(3)")
            gb.configure_pagination(enabled=False)
            gb.configure_grid_options(domLayout="normal")
            grid_options = gb.build()

            grid = AgGrid(
                grid_df,
                gridOptions=grid_options,
                height=TABLE_HEIGHT,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                fit_columns_on_grid_load=False,
                theme="streamlit",
                key="alarm_grid",
            )

            selected_rows = grid.get("selected_rows", [])

            if isinstance(selected_rows, pd.DataFrame):
                first = selected_rows.iloc[0].to_dict() if not selected_rows.empty else None
            elif isinstance(selected_rows, list):
                first = selected_rows[0] if len(selected_rows) > 0 else None
            else:
                first = None

            if first is not None:
                ev_id = first.get("event_id")
                if ev_id and ev_id != st.session_state.selected_alarm_event_id:
                    st.session_state.selected_alarm_event_id = ev_id
                    row = alarms.loc[alarms["event_id"] == ev_id].iloc[0].to_dict()
                    st.session_state.selected_event = row
                    st.session_state.selected_sensor = row["sensor_raw"]

                    # ✅ 핵심: 새 알람으로 상세 진입 시, 상세 필터 상태 리셋 (이전 상세 선택이 sensor_raw를 덮어쓰는 버그 방지)
                    st.session_state.detail_stage_one = None
                    st.session_state.detail_type_one = None
                    st.session_state.detail_prev_stage = None
                    st.session_state.detail_prev_type = None

                    st.session_state.page = "sensor"
                    st.rerun()

else:
    # ============================
    # DETAIL
    # ============================
    # ✅ 상세 CSS 블록 복구(스크롤 숨김)
    st.markdown(
        """
    <style>
      html, body { height: 100%; overflow: hidden; }
      section.main { height: 100vh; overflow: hidden; }
      div.block-container { height: 100vh; overflow: hidden; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    evt = st.session_state.get("selected_event")
    sensor_raw = st.session_state.get("selected_sensor")
    if not evt or not sensor_raw:
        st.session_state.page = "main"
        st.rerun()

    cur_stage = parse_process(sensor_raw)
    cur_suffix = parse_suffix(sensor_raw) or ""
    cur_type_ko = SENSOR_SUFFIX_TO_KO.get(cur_suffix, SENSOR_TYPE_OPTIONS[0])

    if st.session_state.detail_stage_one is None:
        st.session_state.detail_stage_one = cur_stage
    if st.session_state.detail_type_one is None:
        st.session_state.detail_type_one = cur_type_ko

    top_l, top_r = st.columns([0.6, 9.4], gap="small")

    with top_l:
        st.markdown('<div class="square-btn">', unsafe_allow_html=True)
        if st.button("←", use_container_width=True):
            st.session_state.page = "main"
            st.session_state.selected_event = None
            st.session_state.selected_sensor = None
            st.session_state.selected_alarm_event_id = None

            # ✅ 상세 필터도 같이 초기화
            st.session_state.detail_stage_one = None
            st.session_state.detail_type_one = None
            st.session_state.detail_prev_stage = None
            st.session_state.detail_prev_type = None

            if "alarm_grid" in st.session_state:
                del st.session_state["alarm_grid"]
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with top_r:
        spacer, filter_col = st.columns([8.3, 1.7], gap="small")
        with filter_col:
            st.markdown('<div class="compact-popover">', unsafe_allow_html=True)
            with st.popover("조건", use_container_width=True):
                new_stage = st.selectbox(
                    "공정 단계",
                    options=all_processes,
                    index=all_processes.index(st.session_state.detail_stage_one)
                    if st.session_state.detail_stage_one in all_processes
                    else 0,
                )
                new_type = st.selectbox(
                    "측정 항목",
                    options=SENSOR_TYPE_OPTIONS,
                    index=SENSOR_TYPE_OPTIONS.index(st.session_state.detail_type_one)
                    if st.session_state.detail_type_one in SENSOR_TYPE_OPTIONS
                    else 0,
                )

                prev_s = st.session_state.detail_prev_stage
                prev_t = st.session_state.detail_prev_type
                if prev_s is not None and prev_t is not None and (new_stage != prev_s or new_type != prev_t):
                    st.toast("조회 조건이 변경되었습니다.", icon="ℹ️")

                st.session_state.detail_stage_one = new_stage
                st.session_state.detail_type_one = new_type
                st.session_state.detail_prev_stage = new_stage
                st.session_state.detail_prev_type = new_type
            st.markdown("</div>", unsafe_allow_html=True)

    stage_pick = st.session_state.detail_stage_one
    type_pick = st.session_state.detail_type_one

    stage_n = stage_to_int(stage_pick)
    suffix = KO_TO_SUFFIX.get(type_pick)
    candidate = f"stage{stage_n}_{suffix}" if (stage_n is not None and suffix) else None
    if candidate and candidate in sensor_cols:
        sensor_raw = candidate
        st.session_state.selected_sensor = sensor_raw

    y_title = sensor_display_name(sensor_raw)
    st.markdown(f"## 센서 관리도 상세 | {parse_process(sensor_raw)} · {y_title}")
    st.divider()

    lim = limits_by_sensor[sensor_raw]
    v = df[sensor_raw].to_numpy(dtype=float)

    ooc_all = (
        np.isfinite(v)
        & np.isfinite(lim["ucl"])
        & np.isfinite(lim["lcl"])
        & ((v > lim["ucl"]) | (v < lim["lcl"]))
    )
    flag_all = np.where(ooc_all, "OOC", "OK")

    center = int(evt.get("order", 0))
    sel_pid = str(evt.get("제품 ID", ""))

    before_n = int(st.session_state.detail_before_n)
    after_n = int(st.session_state.detail_after_n)

    lo = max(0, center - before_n)
    hi = min(len(df) - 1, center + after_n)

    df_plot = pd.DataFrame(
        {
            "order": df.loc[lo:hi, "order"].to_numpy(),
            "id": df.loc[lo:hi, "id"].to_numpy(),
            "value": v[lo : hi + 1],
            "flag": flag_all[lo : hi + 1],
        }
    )

    ooc_in_view = int((df_plot["flag"] == "OOC").sum())

    st.markdown(
        f"""
    <div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">구간 내 한계 이탈 건수</div>
        <div class="kpi-value">{ooc_in_view:,}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">상한관리한계(UCL)</div>
        <div class="kpi-value">{("-" if not np.isfinite(lim["ucl"]) else f"{lim['ucl']:.4f}")}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">하한관리한계(LCL)</div>
        <div class="kpi-value">{("-" if not np.isfinite(lim["lcl"]) else f"{lim['lcl']:.4f}")}</div>
    </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    gf_l, gf_r = st.columns([8.6, 1.4], gap="small")
    with gf_r:
        st.markdown('<div class="compact-popover">', unsafe_allow_html=True)
        with st.popover("표시 구간", use_container_width=True):
            b = st.number_input("이전 샘플 수", min_value=10, max_value=5000, value=before_n, step=10)
            a = st.number_input("이후 샘플 수", min_value=10, max_value=5000, value=after_n, step=10)
            st.session_state.detail_before_n = int(b)
            st.session_state.detail_after_n = int(a)
        st.markdown("</div>", unsafe_allow_html=True)

    # ✅ Plotly 인터랙티브 차트(줌/팬/툴팁) - Altair/Vega 충돌 우회
    x = df_plot["order"].to_numpy()
    y = df_plot["value"].to_numpy()
    ids = df_plot["id"].astype(str).to_numpy()
    flags = df_plot["flag"].astype(str).to_numpy()

    fig = go.Figure()

    # 전체 라인 (hover에서 order 제거)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=y_title,
            customdata=np.column_stack([ids, flags]),
            hovertemplate=(
                f"{y_title}=%{{y:.4f}}<br>"
                "제품 ID=%{customdata[0]}<br>"
                "상태=%{customdata[1]}<extra></extra>"
            ),
        )
    )

    # 한계 이탈 점(가시성 강화: 크기↑ + 색 지정 + 테두리)
    ooc_mask = (df_plot["flag"] == "OOC").to_numpy()
    if np.any(ooc_mask):
        fig.add_trace(
            go.Scatter(
                x=x[ooc_mask],
                y=y[ooc_mask],
                mode="markers",
                name="한계 이탈",
                marker=dict(
                    size=10,
                    color="magenta",
                    line=dict(width=1.5, color="white"),
                    symbol="circle"
                ),
                customdata=ids[ooc_mask],
                hovertemplate=(
                    "한계 이탈<br>"
                    f"{y_title}=%{{y:.4f}}<br>"
                    "제품 ID=%{customdata}<extra></extra>"
                ),
            )
        )

    # 평균/UCL/LCL 수평선
    if np.isfinite(lim.get("mean", np.nan)):
        fig.add_hline(y=float(lim["mean"]))
    if np.isfinite(lim.get("ucl", np.nan)):
        fig.add_hline(y=float(lim["ucl"]))
    if np.isfinite(lim.get("lcl", np.nan)):
        fig.add_hline(y=float(lim["lcl"]))

    # 선택 알람 강조: 세로선 + "빨간 점" (hover에서 order 제거)
    fig.add_vline(x=float(center), line_dash="dash")
    sel = df_plot.loc[df_plot["order"] == center]
    if not sel.empty and np.isfinite(sel["value"].iloc[0]):
        fig.add_trace(
            go.Scatter(
                x=[float(center)],
                y=[float(sel["value"].iloc[0])],
                mode="markers",
                name="선택 제품",
                marker=dict(
                    size=14,
                    color="red",
                    line=dict(width=1.5, color="white"),
                    symbol="circle"
                ),
                customdata=[sel_pid],
                hovertemplate=(
                    "선택 제품<br>"
                    f"{y_title}=%{{y:.4f}}<br>"
                    "제품 ID=%{customdata}<extra></extra>"
                ),
            )
        )

    # ✅ x축 자체 숨김(눈에 보이는 숫자/눈금/축 라인 전부 제거)
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title=y_title,
        xaxis=dict(visible=False, showticklabels=False, showgrid=False, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})