import streamlit as st

st.set_page_config(page_title="Page 3", layout="wide")
st.title("Page 3 - 설정/가이드")

st.write("- 메인: PCA 기반 T²/SPE 관리도 (정상(label=0) 기준으로 리밋 산정)")
st.write("- Page 2: 클릭한 id 상세(원본 레코드 + stage 프로파일 + deviation top)")
st.write("- 필요하면 여기서 리밋 정책(α), PCA 컴포넌트 수, Phase I 기간 정의 등을 고정/설정할 수 있음")