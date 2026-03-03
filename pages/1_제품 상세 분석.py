import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from catboost import CatBoostClassifier
import shap
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT_DIR = HERE.parents[1] if HERE.parent.name == "pages" else HERE.parent

MODEL_PATH = str(ROOT_DIR / "catboost_final_model.cbm")
DATA_PATH  = str(ROOT_DIR / "mice_final_data_with_id.csv")

st.set_page_config(layout="wide", page_title="QA Analysis")

# --- CSS 스타일 수정 (여기가 포인트입니다) ---
st.markdown("""
    <style>
    .block-container {padding-top: 4rem !important; padding-bottom: 0rem !important;}
    .section-title-main { font-size: 19px; font-weight: bold; color: #333; margin: 0; }
    .reason-card {
        background-color: #ffffff; 
        padding: 12px 18px; 
        border-radius: 4px; 
        box-shadow: 1px 1px 4px rgba(0,0,0,0.05);
        border: 1px solid #eee;
        border-left: 1px solid #ccc; 
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        return None, None, None, None, "파일을 찾을 수 없습니다."
    
    try:
        model = CatBoostClassifier().load_model(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        
        # 1. 모델이 요구하는 파생변수 5개 동일하게 계산
        # '차이의 차이' 변수
        df['viscosity_deviation_diff_3_2'] = df.get('stage3_viscosity_deviation', 0) - df.get('stage2_viscosity_deviation', 0)
        df['flow_deviation_diff_5_4'] = df.get('stage5_flow_deviation', 0) - df.get('stage4_flow_deviation', 0)
        df['viscosity_deviation_diff_5_1'] = df.get('stage5_viscosity_deviation', 0) - df.get('stage1_viscosity_deviation', 0)
        df['co2_deviation_diff_5_1'] = df.get('stage5_co2_deviation', 0) - df.get('stage1_co2_deviation', 0)

        # '기울기' 변수
        n_cols = [f'stage{i}_n_deviation' for i in range(1, 6)]
        if all(c in df.columns for c in n_cols):
            df['n_deviation_slope'] = np.polyfit(range(5), df[n_cols].values.T, 1)[0]
        else:
            df['n_deviation_slope'] = 0

        # 2. 전처리 (숫자형 변환 및 결측치)
        for c in df.columns:
            if c != 'id':
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        # 3. 모델 피처 순서 맞추기
        model_features = model.feature_names_
        for col in model_features:
            if col not in df.columns:
                df[col] = 0
        
        # 4. SPC 차트용 통계치 (정상 데이터 기준)
        normal_df = df[df['label'] == 0] if 'label' in df.columns else df
        stats = {
            'mean': normal_df.mean(numeric_only=True),
            'std': normal_df.std(numeric_only=True)
        }

        return df, model, model_features, stats, None
        
    except Exception as e:
        return None, None, None, None, str(e)

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

# 함수 실행 및 결과 할당
df, model, model_features, stats, error_msg = load_all()

if error_msg:
    st.error(f"데이터 로드 오류: {error_msg}")
    st.stop()

explainer = get_explainer(model)

def translate_feature(col):
    # (1) 특수 파생변수 강제 매핑
    manual_map = {
        'viscosity_deviation_diff_3_2': 'S3-S2 점도편차 차이',
        'flow_deviation_diff_5_4': 'S5-S4 유량편차 차이',
        'viscosity_deviation_diff_5_1': 'S5-S1 점도편차 차이',
        'co2_deviation_diff_5_1': 'S5-S1 CO2편차 차이',
        'n_deviation_slope': 'N2 편차 기울기'
    }
    if col in manual_map:
        return manual_map[col]
        
    # (2) 일반 스테이지 변수 처리
    base_sensor_dict = {
        'temp': '온도', 'humidity': '습도', 'flow': '유량', 'density': '밀도',
        'viscosity': '점도', 'co2': 'CO2', 'o2': 'O2', 'n': 'N2',
        'flow_deviation': '유량 편차', 'density_deviation': '밀도 편차',
        'viscosity_deviation': '점도 편차', 'co2_deviation': 'CO2 편차',
        'o2_deviation': 'O2 편차', 'n_deviation': 'N2 편차'
    }
    
    # 단순 센서명 변환 (SPC 리스트용)
    if col in base_sensor_dict:
        return base_sensor_dict[col]

    if col.startswith('stage'):
        parts = col.split('_')
        stage_num = parts[0].replace('stage', 'S')
        sensor_eng = "_".join(parts[1:])
        sensor_kor = base_sensor_dict.get(sensor_eng, sensor_eng)
        return f"{stage_num} {sensor_kor}"
    
    return col.replace('_', ' ')

# --- ID 동기화 로직 ---
id_list = sorted(df['id'].unique().tolist(), reverse=True)
if "target_id" not in st.session_state:
    st.session_state["target_id"] = id_list[0]

# --- 1. 헤더 섹션 ---
col_t, col_f = st.columns([4, 1])
with col_t:
    st.markdown('<div class="section-title-main">집중 관리 후보 항목 TOP 3</div>', unsafe_allow_html=True)

with col_f:
    target_id = st.selectbox("", id_list, key="target_id", label_visibility="collapsed")

# --- 데이터 연산 ---
selected_row = df[df['id'] == target_id].iloc[0]
input_df = pd.DataFrame([{f: (selected_row[f] if f in selected_row else 0.0) for f in model_features}])[model_features]

model_pred = model.predict(input_df)[0]
fail_prob = model.predict_proba(input_df)[0][1] * 100
res_color = "#FF4B4B" if model_pred == 1 else "#00CC96"

shap_vals = explainer.shap_values(input_df)
top_idxs = np.argsort(shap_vals[0])[-3:][::-1]

# --- 1. 점검 메시지 데이터베이스 정의 ---
check_msgs = {
    '온도': {'High': '히터 가열 과다 및 냉각 라인 점검', 'Low': '히터 단선 및 예열 시간 부족 점검'},
    '습도': {'High': '외기 유입 및 배기(Exhaust) 성능 점검', 'Low': '가습 장치 작동 및 밀폐 상태 점검'},
    '유량': {'High': '공급 펌프 압력 및 배관 누수 점검', 'Low': '노즐 막힘 및 필터 교체 주기 점검'},
    '점도': {'High': '약액 농축 및 증발 여부 점검', 'Low': '희석액(DIW) 과잉 혼합 상태 점검'},
    '밀도': {'High': '약액 혼합비 오류 및 기포(Bubble) 점검', 'Low': '약액 소모(Degradation) 및 농도 점검'},
    'O2': {'High': '가스 라인 리크(Leak) 및 탈기 장치 점검', 'Low': '질소 퍼지 라인 및 가스 공급압 점검'},
    'N2': {'High': '레귤레이터 오작동 및 공급 과부하 점검', 'Low': '메인 가스 탱크 잔량 및 라인 폐쇄 점검'},
    'CO2': {'High': '버블러(Bubbler) 과잉 주입 여부 점검', 'Low': 'CO2 인젝터 노즐 및 실린더 점검'}
}

# --- 2. TOP 3 카드 섹션 (PASS/NG 분기 처리) ---
if model_pred == 0:  # 판정이 PASS인 경우
    st.success("✅ **불량 위험이 낮습니다** ")
    st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True) # 여백
else:
    # 판정이 NG(불량)인 경우에만 TOP 3 카드를 보여줌
    c1, c2, c3 = st.columns(3)

    for i, idx in enumerate(top_idxs):
        eng_name = model_features[idx]
        kor_name = translate_feature(eng_name)
        diff = input_df.iloc[0, idx] - stats['mean'].get(eng_name, 0)
        
        status = "Normal"
        if diff > 0.01:
            color, arrow, status = "#FF4B4B", "▲", "High"
        elif diff < -0.01:
            color, arrow, status = "#1f77b4", "▼", "Low"
        else:
            color, arrow = "#95A5A6", "—"

        action_msg = "해당 센서 및 연결 라인 상태 확인" # 기본값
        for key in check_msgs.keys():
            if key in kor_name: # 예: "S3 온도"에 "온도"가 있는지 확인
                action_msg = check_msgs[key].get(status, action_msg)
                break

        with [c1, c2, c3][i]:
            st.markdown(
                f'''
                <div class="reason-card" style="border-left-color: {color};">
                    <b style="font-size: 11px; color: #888; text-transform: uppercase;">RANKING {i+1}</b><br/>
                    <b style="font-size: 17px; color: #111; line-height: 1.4;">{kor_name}</b><br/>
                    <span style="font-size: 15px; color: {color}; font-weight: bold;">
                        정상 대비 {diff:+.2f} {arrow}
                    </span><br/>
                    <div style="margin-top: 2px; border-top: 1px solid #f8f8f8; padding-top: 2px;">
                        <span style="font-size: 12px; color: #333; font-weight: bold;">점검항목:</span>
                        <span style="font-size: 12px; color: #333;"> {action_msg}</span>
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )
# --- 4. SPC 추세 및 판정 결과 ---
title_col, result_col = st.columns([7, 3])

with title_col:
    st.markdown(
        '<p style="font-size:18px; font-weight:bold; margin-top:10px; margin-bottom:5px;">'
        '정상 대비 변수별 추세 분석</p>', 
        unsafe_allow_html=True
    )

with result_col:
    st.markdown(
        f'''
        <div style="text-align: right; margin-top: 10px;">
            <span style="font-size: 16px;"><b>최종 판정:</b> 
                <span style="color:{res_color}; font-weight:bold;">
                    {"불량" if model_pred == 1 else "정상"} ({fail_prob:.1f}%)
                </span>
            </span>
        </div>
        ''', 
        unsafe_allow_html=True
    )

# --- SPC 차트 리스트 ---
v_list = ['temp', 'humidity', 'flow_deviation', 'density_deviation', 'viscosity_deviation', 'co2_deviation', 'o2_deviation', 'n_deviation']
v_map = {
    'temp': '온도', 'humidity': '습도', 
    'flow_deviation': '유량 편차', 'density_deviation': '밀도 편차', 
    'viscosity_deviation': '점도 편차', 'co2_deviation': 'CO2 편차', 
    'o2_deviation': 'O2 편차', 'n_deviation': 'N2 편차'
}

# --- 1. TOP 3 리스트 추출 ---
top_3_list = [model_features[idx] for idx in top_idxs]

# --- 2. SPC 차트 그리드 생성 ---
for r in range(2):
    grid = st.columns(4)
    for c in range(4):
        idx = r * 4 + c
        if idx >= len(v_list): break
        v_name = v_list[idx]
        kor_v_name = v_map.get(v_name, v_name)
        
        stages = ["S1", "S2", "S3", "S4", "S5"]
        y_curr = [selected_row.get(f"stage{j}_{v_name}", 0) for j in range(1, 6)]
        y_mean = [stats['mean'].get(f"stage{j}_{v_name}", 0) for j in range(1, 6)]
        y_std = [stats['std'].get(f"stage{j}_{v_name}", 0) for j in range(1, 6)]

        # --- [추가] Y축 범위 최적화 로직 ---
        avg_std = sum(y_std) / 5
        avg_mean = sum(y_mean) / 5
        view_margin = avg_std * 4 if avg_std > 0 else abs(avg_mean) * 0.1
        
        all_points = y_curr + y_mean
        final_min = min(min(all_points), avg_mean - view_margin)
        final_max = max(max(all_points), avg_mean + view_margin)
        pad = (final_max - final_min) * 0.05

        # --- 스타일 및 데이터 준비 ---
        line_color = "#6e6e6e"
        line_width = 2
        m_colors = ['#6e6e6e'] * 5
        highlight_paths = [] 

        if model_pred == 1: # NG일 때만 강조
            for tf in top_3_list:
                if v_name in tf:
                    if "slope" in tf:
                        line_color = "#FF4B4B"; line_width = 3
                        m_colors = ["#FF4B4B"] * 5
                    elif "_diff_" in tf:
                        try:
                            parts = tf.split('_')
                            s_start_idx = min(int(parts[-1]), int(parts[-2])) - 1
                            s_end_idx = max(int(parts[-1]), int(parts[-2])) - 1
                            m_colors[s_start_idx] = "#FF4B4B"
                            m_colors[s_end_idx] = "#FF4B4B"
                            highlight_paths.append({'x': stages[s_start_idx:s_end_idx+1], 'y': y_curr[s_start_idx:s_end_idx+1]})
                        except: pass
                    elif "stage" in tf:
                        try:
                            s_idx = int(tf.split('_')[0].replace('stage', '')) - 1
                            m_colors[s_idx] = "#FF4B4B"
                        except: pass

        with grid[c]:
            fig = go.Figure()
            
            # 1. 배경 평균 점선
            fig.add_trace(go.Scatter(x=stages, y=y_mean, mode='lines', 
                                     line=dict(color='rgba(46, 204, 113, 0.4)', width=2, dash='dash')))
            
            # 2. 메인 데이터 선 (회색 혹은 Slope일 때 빨강)
            fig.add_trace(go.Scatter(x=stages, y=y_curr, mode='lines', 
                                     line=dict(color=line_color, width=line_width)))
            
            # 3. Diff 구간 빨간 선 덧그리기 (경로 추적)
            if model_pred == 1 and highlight_paths:
                for path in highlight_paths:
                    fig.add_trace(go.Scatter(x=path['x'], y=path['y'], mode='lines', 
                                             line=dict(color="#FF4B4B", width=2), hoverinfo='skip'))

            # 4. 마커를 최상단에 (회색 점이 빨간 선 위로 올라오도록)
            fig.add_trace(go.Scatter(
                x=stages, y=y_curr, 
                mode='markers', 
                marker=dict(size=8, color=m_colors, line=dict(width=1, color='white')),
                hoverinfo='skip'
            ))

            fig.update_layout(
                title=dict(text=f"<b>{kor_v_name}</b>", font=dict(size=12)),
                height=180, margin=dict(l=10, r=10, t=35, b=10),
                template="plotly_white", showlegend=False,
                yaxis=dict(tickfont=dict(size=9), range=[final_min - pad, final_max + pad], zeroline=False),
                xaxis=dict(tickfont=dict(size=9))
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})