import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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
# - [업그레이드] 슬라이더 가드레일, Top 3 위험 요인 분석, AI 최적화 버튼 적용
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
      
      /* AI 버튼 스타일 */
      .ai-btn-container button {{
          width: 100%; font-weight: 800; background-color: #1428A0; color: white;
          border-radius: 8px; border: none; padding: 10px 0;
      }}
      .ai-btn-container button:hover {{ background-color: #0F1D7A; color: white; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# 8대 핵심 지표 한글 매핑
# ----------------------------
SENSOR_KO = {
    "temp": "온도", 
    "humidity": "습도", 
    "flow_deviation": "유량 편차", 
    "density_deviation": "밀도 편차", 
    "viscosity_deviation": "점도 편차", 
    "o2_deviation": "O2 편차", 
    "n_deviation": "N2 편차", 
    "co2_deviation": "CO2 편차",
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

# 각 핵심 지표별 통계 도출 (가드레일용 Q1, Q3 추가)
global_stats = {}
for metric in CORE_METRICS:
    related_cols = [c for c in df.columns if c.endswith(f"_{metric}") and c.startswith("stage")]
    if related_cols:
        vals = df[related_cols].values.flatten()
        vals = vals[~np.isnan(vals)] 
        global_stats[metric] = {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "median": float(np.median(vals)),
            "q1": float(np.percentile(vals, 25)),
            "q3": float(np.percentile(vals, 75))
        }

# 세션 상태 초기화 (슬라이더 조작용)
for metric in CORE_METRICS:
    if f"slider_{metric}" not in st.session_state:
        st.session_state[f"slider_{metric}"] = global_stats.get(metric, {}).get("median", 0.0)

# AI 최적화 버튼 콜백 함수
def reset_to_optimal():
    for m in CORE_METRICS:
        if m in global_stats:
            st.session_state[f"slider_{m}"] = global_stats[m]["median"]

# ----------------------------
# Inference Data Generator
# ----------------------------
def generate_inference_row(master_state, required_features):
    row = {}
    for feat in required_features:
        if feat.startswith("stage"):
            parts = feat.split("_", 1)
            if len(parts) == 2 and parts[1] in master_state:
                row[feat] = master_state[parts[1]]
            else:
                row[feat] = 0.0
        elif "_diff_" in feat or "_slope" in feat:
            row[feat] = 0.0
        else:
            row[feat] = 0.0
    return row

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="topbar"><div class="brand">SPARTA ELECTRONICS · AI 미래 품질 예측</div><div class="subtitle">핵심 8대 공정 변수 조작 및 실시간 불량 확률 시뮬레이션</div></div>', unsafe_allow_html=True)

# ----------------------------
# Main Layout
# ----------------------------
left_col, right_col = st.columns([1.2, 1.8], gap="large")

with left_col:
    st.markdown('<div class="panel"><div class="pt">8대 지표 컨트롤</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:13px; color:#6B7280; margin-bottom:20px; line-height:1.5;">아래 조절바를 움직이면 1~5공정 전체에 해당 값이 일괄 적용되어 AI가 즉각적으로 불량 확률을 예측합니다.</p>', unsafe_allow_html=True)
    
    # 1. AI 최적화 레시피 리셋 버튼 추가
    st.markdown('<div class="ai-btn-container">', unsafe_allow_html=True)
    st.button("AI 최적 공정 조건 자동 세팅", on_click=reset_to_optimal, use_container_width=True)
    st.markdown('</div><div style="height: 15px;"></div>', unsafe_allow_html=True)

    master_inputs = {}
    
    # 2. 슬라이더에 가드레일(권장 범위) 추가 렌더링
    for metric in CORE_METRICS:
        if metric in global_stats:
            kor_name = SENSOR_KO.get(metric, metric)
            min_v = global_stats[metric]["min"]
            max_v = global_stats[metric]["max"]
            med_v = global_stats[metric]["median"]
            q1_v = global_stats[metric]["q1"]
            q3_v = global_stats[metric]["q3"]
            
            # 가드레일 텍스트 추가
            label_text = f"{kor_name} (권장: {q1_v:.2f} ~ {q3_v:.2f})"
            
            master_inputs[metric] = st.slider(
                label=label_text,
                min_value=min_v,
                max_value=max_v,
                step=(max_v - min_v) / 100.0 if max_v > min_v else 0.1,
                key=f"slider_{metric}"
            )
            st.markdown('<div style="height: 5px;"></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    inference_row = generate_inference_row(master_inputs, model_features)
    sim_df = pd.DataFrame([inference_row])[model_features] 
    
    pred_proba = model.predict_proba(sim_df)[0]
    defect_prob = pred_proba[1] * 100 
    
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
        
        st.markdown(f'''
        <div style="background-color:#F3F4F6; padding:10px; border-radius:8px; border:1px solid #E5E7EB;">
            <div style="font-size:12px; font-weight:800; color:#374151; margin-bottom:4px;">운영 임계값 (Threshold) 기준</div>
            <div style="font-size:11px; color:#6B7280;">- <b>정상:</b> 0% ~ 51.48% 미만</div>
            <div style="font-size:11px; color:#6B7280;">- <b>주의:</b> 51.48% ~ 80% 미만</div>
            <div style="font-size:11px; color:#6B7280;">- <b>위험:</b> 80% 이상</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 10px 0; border-color: #E5E7EB;'>", unsafe_allow_html=True)
    
    # 3. 실시간 위험 요인 Top 3 분석 로직
    deviations = []
    for metric, current_val in master_inputs.items():
        if metric in global_stats:
            med_v = global_stats[metric]["median"]
            min_v = global_stats[metric]["min"]
            max_v = global_stats[metric]["max"]
            
            # 중앙값 대비 어느 정도(비율) 틀어졌는지 계산
            range_span = max_v - min_v if max_v > min_v else 1.0
            deviation_score = abs(current_val - med_v) / range_span
            
            deviations.append({
                "name": SENSOR_KO.get(metric, metric).split(" ")[0],
                "score": deviation_score,
                "current": current_val,
                "diff": current_val - med_v
            })
            
    deviations.sort(key=lambda x: x["score"], reverse=True)
    top3_risks = deviations[:3]

    st.markdown('<div style="font-size:15px; font-weight:900; color:#111827; margin-bottom:12px;">실시간 위험 요인 Top 3 분석 (기준값 대비 이탈률)</div>', unsafe_allow_html=True)
    
    risk_html = ""
    for idx, risk in enumerate(top3_risks):
        direction = "초과" if risk["diff"] > 0 else "미달"
        color = BAD if risk["score"] > 0.2 else (WARN if risk["score"] > 0.1 else GOOD)
        
        risk_html += f'''
        <div style="display:flex; justify-content:space-between; align-items:center; background:#F9FAFB; padding:12px 16px; border-radius:8px; margin-bottom:8px; border-left: 4px solid {color}; border-top: 1px solid #E5E7EB; border-right: 1px solid #E5E7EB; border-bottom: 1px solid #E5E7EB;">
            <div style="font-size:14px; font-weight:800; color:#374151;">{idx+1}위. {risk['name']}</div>
            <div style="font-size:13px; font-weight:700; color:{color};">기준치 대비 {abs(risk['diff']):.2f} {direction}</div>
        </div>
        '''
    st.markdown(risk_html, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
