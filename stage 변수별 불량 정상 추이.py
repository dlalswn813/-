import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.covariance import LedoitWolf

DATA_PATH = "C:/Users/blue1/OneDrive/바탕 화면/실전 프로젝트/mice_final_data_with_id.csv"

@st.cache_data
def load_long(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index":"id"})
    pattern = re.compile(r"^stage(\d+)_(.+)$")
    feat_cols = [c for c in df.columns if pattern.match(c)]
    long_df = df.melt(id_vars=["id","label"], value_vars=feat_cols,
                      var_name="feature", value_name="value")
    long_df["stage"] = long_df["feature"].str.extract(r"^stage(\d+)_").astype(int)
    long_df["metric"] = long_df["feature"].str.replace(r"^stage\d+_", "", regex=True)
    long_df["id_num"] = pd.to_numeric(long_df["id"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    return long_df.dropna(subset=["id_num","value","label"])

long_df = load_long(DATA_PATH)

# --- 레이아웃: 왼쪽 세로 탭(변수), 오른쪽 그래프 영역 ---
left_menu, main = st.columns([1, 3], gap="large")

# 가로 Stage 탭(성능 이슈 있으면 st.radio(horizontal=True)로 바꿔)
with main:
    stage_tabs = st.tabs([f"Stage {i}" for i in range(1, 6)])

# 세로 탭(변수): 왼쪽 컬럼에 radio로 구현
metrics = sorted(long_df["metric"].unique().tolist())
metric_sel = left_menu.radio("변수 선택(세로 탭)", metrics)

# bin size
bin_size = left_menu.slider("추이 구간 크기(bin)", 50, 1000, 200, 50)
agg_func = left_menu.selectbox("집계", ["mean", "median"])

def aggregate_trend(df, bin_size, agg_func):
    df = df.sort_values("id_num").copy()
    # bin index
    df["bin"] = (np.arange(len(df)) // bin_size).astype(int)
    df["x"] = df.groupby("bin")["id_num"].transform("median")  # bin 대표 x

    if agg_func == "median":
        g = df.groupby(["bin","label"]).agg(x=("x","median"), y=("value","median")).reset_index()
    else:
        g = df.groupby(["bin","label"]).agg(x=("x","median"), y=("value","mean")).reset_index()
    return g

def plot_two_lines(g, title):
    fig = go.Figure()
    for lab, name in [(0, "정상"), (1, "불량")]:
        gg = g[g["label"] == lab]
        fig.add_trace(go.Scatter(
            x=gg["x"], y=gg["y"],
            mode="lines+markers",
            name=name
        ))
    fig.update_layout(
        height=450,
        template="plotly_white",
        title=title,
        xaxis_title="id(숫자 변환, bin 대표값)",
        yaxis_title="value"
    )
    return fig

# Stage 탭별로 “그 탭이 선택됐을 때만” 그리려면 tabs 대신 stage radio가 더 깔끔함.
# 여기서는 tabs 구조를 유지하면서, 각 탭 안에서 stage만 바꿔 처리.
for s, tab in enumerate(stage_tabs, start=1):
    with tab:
        d = long_df[(long_df["stage"] == s) & (long_df["metric"] == metric_sel)].copy()

        # 정상/불량 둘 중 하나가 너무 적으면 안내
        if d[d["label"] == 0].empty or d[d["label"] == 1].empty:
            st.warning("정상/불량 중 한쪽 데이터가 부족해서 2개 선 비교가 어려움.")
            st.dataframe(d[["id","label","value"]].head(20))
            continue

        g = aggregate_trend(d, bin_size=bin_size, agg_func=agg_func)
        fig = plot_two_lines(g, title=f"Stage {s} / {metric_sel} : 정상 vs 불량 (추이형 집계)")
        st.plotly_chart(fig, use_container_width=True)