import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Page 1", layout="wide")

DATA_PATH = "C:/Users/blue1/OneDrive/바탕 화면/실전 프로젝트/mice_final_data_with_id.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

st.title("Page 1 - Stage 프로파일(선택 id 기준)")

df = load_data()
sid = st.session_state.get("selected_id")

if not sid:
    st.info("메인에서 점을 클릭하면 선택 id가 전달됩니다. 또는 아래에서 id를 직접 고르세요.")
    sid = st.selectbox("id 선택", df["id"].unique())

row = df.loc[df["id"] == sid].iloc[0]
st.write(f"선택 id: **{sid}**, label: **{int(row['label'])}**")

temp_cols = [f"stage{i}_temp" for i in range(1, 6)]
hum_cols  = [f"stage{i}_humidity" for i in range(1, 6)]

temp_df = pd.DataFrame({"stage": [1,2,3,4,5], "temp": [row[c] for c in temp_cols]})
hum_df  = pd.DataFrame({"stage": [1,2,3,4,5], "humidity": [row[c] for c in hum_cols]})

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.line(temp_df, x="stage", y="temp", markers=True, title="Stage별 Temp"), use_container_width=True)
with c2:
    st.plotly_chart(px.line(hum_df, x="stage", y="humidity", markers=True, title="Stage별 Humidity"), use_container_width=True)