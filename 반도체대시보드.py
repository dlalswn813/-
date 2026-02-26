import streamlit as st
from datetime import datetime
import hashlib

st.set_page_config(page_title="반도체 대시보드", layout="wide")

st.title("배포 업데이트 체크")
st.write("이 화면이 보이면 코드가 반영된 거야.")
st.write("현재 시각(서버):", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# 깃 커밋을 바꿀 때마다 달라지는 텍스트(수동 버전)
VERSION = "v1"  # 바꿀 때마다 v2, v3로 수정
st.write("버전:", VERSION)

st.divider()