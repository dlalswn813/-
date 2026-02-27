import streamlit as st
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import plotly.express as px
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="반도체 대시보드", layout="wide")

DATA_PATH = "C:/Users/blue1/OneDrive/바탕 화면/실전 프로젝트/mice_final_data_with_id.csv"  # 반도체대시보드 폴더 내에 있다고 가정

st.title("반도체 대시보드 - 메인")

# -----------------------------
# 데이터 로드 (id/label 방어)
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 컬럼명 정리(BOM/공백 제거)
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    # id 복구 (Unnamed/index 케이스)
    if "id" not in df.columns:
        unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
        if unnamed:
            c0 = unnamed[0]
            if df[c0].astype(str).str.match(r"^S\d+$").mean() > 0.8:
                df = df.rename(columns={c0: "id"})

    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})

    # label 복구
    if "label" not in df.columns:
        for cand in ["LABEL", "Label", "y", "target"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "label"})
                break

    required = {"id", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"필수 컬럼 누락: {required} / 현재 컬럼: {df.columns.tolist()}")

    return df


# -----------------------------
# MSPC (PCA 기반 T², SPE) 계산
# -----------------------------
@st.cache_data
def compute_mspc_scores(df: pd.DataFrame, n_components: int = 5, alpha: float = 0.99):
    feature_cols = [c for c in df.columns if c not in ("id", "label")]
    X = df[feature_cols].copy()

    # 숫자화 + 결측 처리
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    # 분산 0 컬럼 제거
    std = X.std(numeric_only=True)
    keep_cols = std[std > 0].index
    X = X.loc[:, keep_cols]

    normal_mask = (df["label"] == 0)
    if int(normal_mask.sum()) < 10:
        raise ValueError("label==0(정상) 샘플이 너무 적어서 리밋 계산 불가")

    scaler = StandardScaler()
    Xn = scaler.fit_transform(X.loc[normal_mask])
    Xa = scaler.transform(X)

    pca = PCA(n_components=min(n_components, Xa.shape[1]), random_state=0)
    pca.fit(Xn)

    scores = pca.transform(Xa)

    # Hotelling T²
    eigvals = pca.explained_variance_
    inv_eig = 1.0 / eigvals
    T2 = (scores ** 2 * inv_eig).sum(axis=1)

    # SPE(Q)
    Xa_hat = pca.inverse_transform(scores)
    resid = Xa - Xa_hat
    SPE = (resid ** 2).sum(axis=1)

    # 정상 기준 분위수 리밋
    T2_lim = float(np.quantile(T2[normal_mask.values], alpha))
    SPE_lim = float(np.quantile(SPE[normal_mask.values], alpha))

    out = df[["id", "label"]].copy()
    out["T2"] = T2
    out["SPE"] = SPE
    out["risk"] = (out["T2"] > T2_lim) | (out["SPE"] > SPE_lim)
    return out, T2_lim, SPE_lim


# -----------------------------
# 클릭 이벤트 콜백
# -----------------------------
def _on_chart_select():
    """
    st.plotly_chart(key="mspc_scatter", on_select="rerun") 결과가
    st.session_state["mspc_scatter"]에 들어온다.
    여기서 선택된 point의 customdata(id)를 꺼내 Page 2로 이동.
    """
    event = st.session_state.get("mspc_scatter")
    if not event:
        return

    points = getattr(event.selection, "points", None)
    if not points:
        return
    if len(points) == 0:
        return

    # px.scatter(custom_data=["id"])를 넣으면, points[0]["customdata"]에 id가 들어옴
    pid = points[0].get("customdata", None)
    if isinstance(pid, (list, tuple)) and len(pid) > 0:
        pid = pid[0]

    if pid:
        st.session_state["selected_id"] = pid
        st.switch_page("pages/Page 2.py")  # pages/ 구조면 "pages/Page 2.py"로 바꿔


# -----------------------------
# 사이드바 설정
# -----------------------------
with st.sidebar:
    st.subheader("MSPC 설정")
    n_comp = st.slider("PCA components", 2, 15, 5, 1)
    alpha = st.slider("Limit quantile (정상 기준)", 0.90, 0.999, 0.99, 0.001)
    st.caption("점(특히 RISK)을 클릭하면 Page 2로 이동")


# -----------------------------
# 실행
# -----------------------------
df = load_data(DATA_PATH)
score_df, T2_lim, SPE_lim = compute_mspc_scores(df, n_components=n_comp, alpha=alpha)

score_df["point_type"] = np.where(score_df["risk"], "RISK", "OK")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(score_df):,}")
c2.metric("T² limit", f"{T2_lim:.3f}")
c3.metric("SPE limit", f"{SPE_lim:.3f}")
c4.metric("RISK count", f"{int(score_df['risk'].sum()):,}")

st.divider()

# Plotly: 클릭으로 point가 "선택"되도록 clickmode 설정 필요
fig = px.scatter(
    score_df,
    x="T2",
    y="SPE",
    color="point_type",
    hover_data=["id", "label"],
    custom_data=["id"],  # 선택된 점에서 id 추출
    title="T² vs SPE (점 클릭 → 상세 Page 2 이동)",
)
fig.add_vline(x=T2_lim, line_dash="dash")
fig.add_hline(y=SPE_lim, line_dash="dash")
fig.update_layout(height=650, clickmode="event+select")  # 중요

# 핵심: streamlit 내장 selection 이벤트 사용
# - on_select="rerun" 으로 클릭 시 리런
# - key로 event를 session_state에서 읽고 callback으로 처리(지연/한 박자 문제 방지) :contentReference[oaicite:1]{index=1}
st.plotly_chart(
    fig,
    use_container_width=True,
    key="mspc_scatter",
    on_select="rerun",
    selection_mode="points",
)

# 클릭 처리 (콜백 방식)
_on_chart_select()

st.info("점이 선택되면 자동으로 Page 2로 이동합니다. (선택이 안 되면 점을 한 번 더 클릭해서 '선택' 상태로 만드세요.)")