import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Page 2 - 제품 상세", layout="wide")

DATA_PATH = "C:/Users/blue1/OneDrive/바탕 화면/실전 프로젝트/mice_final_data_with_id.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

st.title("Page 2 - 제품 상세")

df = load_data()
sid = st.session_state.get("selected_id")

if not sid:
    st.warning("선택된 id가 없습니다. 메인에서 빨간 점(RISK)을 클릭하세요.")
    st.stop()

row = df.loc[df["id"] == sid]
if row.empty:
    st.error(f"id를 찾을 수 없습니다: {sid}")
    st.stop()

r = row.iloc[0]
st.subheader(f"ID: {sid}")
st.write(f"label: **{int(r['label'])}**")

with st.expander("원본 레코드", expanded=True):
    st.dataframe(row, use_container_width=True)

# Stage profile
temp_cols = [f"stage{i}_temp" for i in range(1, 6)]
hum_cols  = [f"stage{i}_humidity" for i in range(1, 6)]

temp_df = pd.DataFrame({"stage": [1,2,3,4,5], "temp": [r[c] for c in temp_cols]})
hum_df  = pd.DataFrame({"stage": [1,2,3,4,5], "humidity": [r[c] for c in hum_cols]})

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.line(temp_df, x="stage", y="temp", markers=True, title="Stage별 Temp"), use_container_width=True)
with c2:
    st.plotly_chart(px.line(hum_df, x="stage", y="humidity", markers=True, title="Stage별 Humidity"), use_container_width=True)

# deviation top
dev_cols = [c for c in df.columns if c.endswith("_deviation")]
dev = pd.Series({c: float(r[c]) for c in dev_cols}).sort_values(key=lambda s: s.abs(), ascending=False).head(15)
top15 = dev.reset_index()
top15.columns = ["variable", "value"]

st.subheader("Deviation Top 15 (절대값 기준)")
st.plotly_chart(px.bar(top15, x="value", y="variable", orientation="h"), use_container_width=True)