import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import re

# 캣부스트 라이브러리 임포트
try:
    from catboost import CatBoostClassifier
except ImportError:
    st.error("CatBoost 라이브러리가 설치되지 않았습니다. 터미널에서 'pip install catboost'를 실행하세요.")
    st.stop()

# =========================================================
# AI Predictive Maintenance Dashboard (What-If Simulator)
# - CatBoost 모델 연동 기반 미래 불량 확률 예측
# - 직관적인 8대 핵심 공정 지표(Master Variable) 조작 UI
# - 51.48%, 80% 임계값(Threshold) 기반 게이지 차트
# =========================================================

st.set_page_config(page_title="AI 불량 예측 시뮬레이터", layout="wide")

# ----------------------------
# Styling & Theme
# ----------------------------
SAMSUNG_BLUE_2 = "#1428A0"
GOOD = "#10B981"
WARN = "#F59E0B"
BAD = "#DC2626"
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
        --text:{TEXT}; --muted:{MUTED}; --blue2:{SAMSUNG_BLUE_2};
      }}
      .stApp {{ background: var(--bg); color: var(--text); }}
      .topbar {{
        background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px;
        padding: 16px 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      }}
      .brand {{ font-size: 24px; font-weight: 900; letter-spacing: 0.4px; color: var(--blue2); }}
      .subtitle {{ color: var(--muted); font-weight: 700; margin-top: 2px; }}
      .panel {{
        background: var(--panel); border: 1px solid var(--stroke); border-radius: 12px;
        padding: 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); height: 100%;
      }}
      .pt {{ font-weight: 900; margin-bottom: 15px; font-size: 18px; border-bottom: 1px solid #E5E7EB; padding-bottom: 10px; }}
      
      /* 슬라이더 라벨(제목) 폰트 강조 */
      div[data-testid="stSlider"] label p {{ font-weight: 800 !important; color: #111827 !important; font-size: 14px !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# 8대 핵심 지표 한글 매핑
# ----------------------------
SENSOR_KO = {
    "temp": "온도 (Temperature)", 
    "humidity": "습도 (Humidity)", 
    "flow_deviation": "유량 편차 (Flow Deviation)", 
    "density_deviation": "밀도 편차 (Density Deviation)", 
    "viscosity_deviation": "점도 편차 (Viscosity Deviation)", 
    "o2_deviation": "O2 편차 (O2 Deviation)", 
    "n_deviation": "N2 편차 (N2 Deviation)", 
    "co2_deviation": "CO2 편차 (CO2 Deviation)",
}

CORE_METRICS = [
    "temp", "humidity", "flow_deviation", "density_deviation",
    "viscosity_deviation", "o2_deviation", "n_deviation", "co2_deviation"
]

# ----------------------------
# Data & Model Load
# ----------------------------
DATA_PATH = "mice_final_data_with_id.csv"
MODEL_PATH = "catboost_final_model.cbm"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"데이터 파일이 없습니다: {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ("id", "label", "_id_sort", "run", "_id_num")]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, feature_cols

@st.cache_resource
def load_ml_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"CatBoost 모델 파일이 없습니다: {MODEL_PATH}")
        st.stop()
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model

df, original_feature_cols = load_data()
model = load_ml_model()

model_features = model.feature_names_ if hasattr(model, 'feature_names_') else original_feature_cols

# 각 핵심 지표별로 전체 공정(stage 1~5) 데이터를 통합하여 최소/최대/중앙값 도출
global_stats = {}
for metric in CORE_METRICS:
    related_cols = [c for c in df.columns if c.endswith(f"_{metric}") and c.startswith("stage")]
    if related_cols:
        vals = df[related_cols].values.flatten()
        vals = vals[~np.isnan(vals)] # 결측치 제거
        global_stats[metric] = {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "median": float(np.median(vals))
        }

# ----------------------------
# Inference Data Generator (동적 할당)
# ----------------------------
def generate_inference_row(master_state, required_features):
    """
    사용자가 8개의 마스터 슬라이더를 조작하면,
    전체 공정(1~5)에 동일한 값을 일괄 세팅하여 파생변수(diff, slope)까지 재계산
    """
    row = {}
    for feat in required_features:
        # 1. Base Feature 처리 (예: stage1_temp)
        if feat.startswith("stage"):
            parts = feat.split("_", 1)
            if len(parts) == 2 and parts[1] in master_state:
                row[feat] = master_state[parts[1]]
            else:
                row[feat] = 0.0
                
        # 2. Diff (차이) 파생변수 처리
        elif "_diff_" in feat:
            # 일괄 세팅하므로 공정간 차이는 0
            row[feat] = 0.0
            
        # 3. Slope (기울기) 파생변수 처리
        elif "_slope" in feat:
            # 일괄 세팅하므로 추세 변화량(기울기)은 0
            row[feat] = 0.0
            
        else:
            row[feat] = 0.0
    return row

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="topbar"><div class="brand">SPARTA ELECTRONICS · AI 미래 품질 예측 (What-If)</div><div class="subtitle">핵심 8대 공정 변수 조작 및 실시간 불량 확률 시뮬레이션</div></div>', unsafe_allow_html=True)

# ----------------------------
# Main Layout
# ----------------------------
left_col, right_col = st.columns([1.2, 1.8], gap="large")

with left_col:
    st.markdown('<div class="panel"><div class="pt">8대 핵심 지표 마스터 컨트롤</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:13px; color:#6B7280; margin-bottom:20px; line-height:1.5;">아래 조절바를 움직이면 1~5공정 전체에 해당 값이 일괄 적용되어 AI가 즉각적으로 불량 확률을 예측합니다.</p>', unsafe_allow_html=True)
    
    master_inputs = {}
    
    # 8개의 마스터 슬라이더 렌더링
    for metric in CORE_METRICS:
        if metric in global_stats:
            kor_name = SENSOR_KO.get(metric, metric)
            min_v = global_stats[metric]["min"]
            max_v = global_stats[metric]["max"]
            med_v = global_stats[metric]["median"]
            
            master_inputs[metric] = st.slider(
                label=kor_name,
                min_value=min_v,
                max_value=max_v,
                value=med_v,
                step=(max_v - min_v) / 100.0 if max_v > min_v else 0.1,
                key=f"slider_{metric}"
            )
            st.markdown('<div style="height: 5px;"></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    # 1. 모델 입력 데이터 실시간 재구축
    inference_row = generate_inference_row(master_inputs, model_features)
    sim_df = pd.DataFrame([inference_row])[model_features] 
    
    # 2. AI 확률 예측 (CatBoost)
    pred_proba = model.predict_proba(sim_df)[0]
    defect_prob = pred_proba[1] * 100  # 불량일 확률 (%)
    
    # 우리가 정한 임계값(Threshold) 로직 적용
    if defect_prob < 51.48:
        gauge_color = GOOD
        status_msg = "안정 (Stable)"
        status_desc = "설정된 데이터는 정상 스펙 범위에 있습니다. 불량 발생 가능성이 낮습니다."
    elif defect_prob < 80.0:
        gauge_color = WARN
        status_msg = "주의 (Warning)"
        status_desc = "위험 임계값(51.48%)을 초과했습니다. 지속 모니터링 및 선제적 밸브 조정이 권장됩니다."
    else:
        gauge_color = BAD
        status_msg = "위험 (Critical)"
        status_desc = "위험 임계값(80.0%)을 초과했습니다. 즉각적인 공정 중단 및 레시피 전면 수정이 필요합니다."

    # 상단: 게이지 차트 패널
    st.markdown('<div class="panel" style="height: auto;"><div class="pt">AI 실시간 예측 결과</div>', unsafe_allow_html=True)
    
    g_col1, g_col2 = st.columns([1.2, 1])
    
    with g_col1:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = defect_prob,
            number = {'suffix': "%", 'font': {'size': 50, 'color': gauge_color, 'weight': 'bold'}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "불량 발생 예상 확률", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': gauge_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 51.48], 'color': "rgba(16, 185, 129, 0.15)"},
                    {'range': [51.48, 80], 'color': "rgba(245, 158, 11, 0.15)"},
                    {'range': [80, 100], 'color': "rgba(220, 38, 38, 0.15)"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': defect_prob}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT))
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
        
    with g_col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: {gauge_color}; margin-bottom: 5px; font-weight: 900;'>{status_msg}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #4B5563; font-size: 15px; line-height: 1.6;'>{status_desc}</p>", unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 20px 0; border-color: #E5E7EB;'>", unsafe_allow_html=True)
        
        # 임계값 안내 범례 추가 (이모지 제거 완료)
        st.markdown(f'''
        <div style="background-color:#F3F4F6; padding:10px; border-radius:8px; border:1px solid #E5E7EB;">
            <div style="font-size:12px; font-weight:800; color:#374151; margin-bottom:4px;">운영 임계값 (Threshold) 기준</div>
            <div style="font-size:11px; color:#6B7280;">- <b>정상:</b> 0% ~ 51.48% 미만</div>
            <div style="font-size:11px; color:#6B7280;">- <b>주의:</b> 51.48% ~ 80% 미만</div>
            <div style="font-size:11px; color:#6B7280;">- <b>위험:</b> 80% 이상</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
