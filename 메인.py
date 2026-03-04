import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from catboost import CatBoostClassifier, Pool
from pathlib import Path

st.set_page_config(page_title="반도체 세정 공정 모니터링", layout="wide")

# CSS 스타일 업데이트 (제목 디자인 및 여백 최적화)
st.markdown("""
    <style>
    /* 1. 전체 페이지 상단 여백 조절 */
    .block-container { 
        padding-top: 3.0rem !important; 
        padding-bottom: 0rem !important; 
    }

    /* 2. 대제목 스타일 및 간격 조절 */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1f2d3d;
        margin-top: 0rem !important;
        margin-bottom: 3px !important; 
        padding-bottom: 0.2rem;
        border-bottom: 2px solid #f0f2f6;
    }

    /* 3. 모든 위젯 사이의 간격을 3px로 압축 */
    [data-testid="stVerticalBlock"] {
        gap: 3px !important;
    }

    /* 4. 구분선 여백 제거 */
    .stDivider {
        margin-top: 0px !important;
        margin-bottom: 0px !important;
    }

    /* 5. KPI 카드 스타일 */
    .kpi-card {
        background-color: #f8f9fa !important;
        padding: 12px !important;
        border-radius: 10px !important;
        border: 1px solid #e0e0e0 !important;
        height: 110px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        align-items: center !important;
        text-align: center !important;
    }
    .kpi-label { font-size: 0.85rem !important; color: #666 !important; font-weight: 500 !important; }
    .kpi-value { font-size: 1.6rem !important; font-weight: 700 !important; color: #1f2d3d !important; line-height: 1 !important; }
    
    .kpi-bottom { height: 18px; display: flex; align-items: center; justify-content: center; }
    .kpi-delta { font-size: 0.85rem; font-weight: 600; }
    .delta-up { color: #ff4b4b !important; }
    .delta-down { color: #00cc66 !important; } 
    .delta-zero { color: #999 !important; }
    .tl-container { display: flex; gap: 6px; justify-content: center; }
    .tl-dot { width: 12px; height: 12px; border-radius: 50%; background-color: #ddd; }
    .tl-red { background-color: #ff4b4b; box-shadow: 0 0 8px #ff4b4b; }
    .tl-yellow { background-color: #ffaa00; box-shadow: 0 0 8px #ffaa00; }
    .tl-green { background-color: #00cc66; box-shadow: 0 0 8px #00cc66; }
            
    /* 6. Streamlit 기본 통지 위젯(st.error 등)의 강제 다이어트 */
    div[data-testid="stNotification"] {
        padding: 0px 10px 5px 10px !important;  /* 내부 여백을 강제로 깎음 */
        margin-top: 0px !important;   /* 위쪽으로 더 끌어올림 */
        margin-bottom: 0px !important;  /* 아래 KPI 카드와의 간격을 제거 */
        min-height: 0px !important;     /* 기본 최소 높이 제한을 무시 */
    }

    /* 상자 안의 텍스트 높이 압축 */
    div[data-testid="stNotification"] h3 {
        font-size: 1.1rem !important; 
        line-height: 1.0 !important; 
        margin: 0px !important;
    }

    </style>
""", unsafe_allow_html=True)

# 1. 제목 추가
st.markdown('<div class="main-title">반도체 세정 공정 실시간 품질 모니터링</div>', unsafe_allow_html=True)

# --- 이후 기존 로직 (MODEL_PATH 정의부터) 동일하게 진행 ---
HERE = Path(__file__).resolve()
ROOT_DIR = HERE.parents[1] if HERE.parent.name == "pages" else HERE.parent

MODEL_PATH = str(ROOT_DIR / "catboost_final_model.cbm")
DATA_PATH  = str(ROOT_DIR / "mice_final_data_with_id.csv")

# ... (이하 생략: 제공해주신 데이터 로드 및 차트 생성 로직 그대로 사용)
def get_feature_kor_map(columns):
    mapping = {}
    for col in columns:
        # 1. [강제 타겟팅] 단어 포함 여부로만 판단 (이게 제일 확실함)
        c = col.lower() # 대소문자 무시
        
        if 'viscosity' in c and 'diff_3_2' in c:
            mapping[col] = 'S3-S2 점도편차 차이'
        elif 'flow' in c and 'diff_5_4' in c:
            mapping[col] = 'S5-S4 유량편차 차이'
        elif 'viscosity' in c and 'diff_5_1' in c:
            mapping[col] = 'S5-S1 점도편차 차이'
        elif 'co2' in c and 'diff_5_1' in c:
            mapping[col] = 'S5-S1 CO2편차 차이'
        elif 'n' in c and 'slope' in c:
            mapping[col] = 'N2 편차 기울기'
        
        # 2. 일반 스테이지 변수 (stage1_flow_deviation 등)
        elif 'stage' in c:
            # 기본 사전
            s_dict = {'temp':'온도', 'humidity':'습도', 'flow':'유량', 'density':'밀도', 
                      'viscosity':'점도', 'co2':'CO2', 'o2':'O2', 'n':'N2'}
            
            # 숫자 추출 (stage1 -> S1)
            import re
            stage_match = re.search(r'stage(\d+)', c)
            s_num = f"S{stage_match.group(1)}" if stage_match else "S?"
            
            # 센서 이름 매칭
            sensor_kor = "변수"
            for eng, kor in s_dict.items():
                if eng in c:
                    sensor_kor = kor
                    break
            
            is_dev = " 편차" if "deviation" in c else ""
            mapping[col] = f"{s_num} {sensor_kor}{is_dev}"
            
        # 3. 그 외 (수동 매핑)
        else:
            manual = {'temp_deviation': '전체 온도 편차', 'hum_deviation': '전체 습도 편차'}
            mapping[col] = manual.get(col, col.replace('_', ' '))
            
    return mapping

def get_h_score(df, latest_row):
    sensor_cols = [c for c in df.columns if any(x in c for x in ['temp', 'humidity', 'deviation'])]
    stats = df[sensor_cols].agg(['mean', 'std']).T
    penalties = 0
    for col in sensor_cols:
        val = latest_row[col]
        mu = stats.loc[col, 'mean']
        sigma = stats.loc[col, 'std']
        z = abs((val - mu) / (sigma + 1e-6))
        if z > 3:
            penalties += (z - 3) * 5
    return max(0, 100 - penalties)

@st.cache_resource(ttl=10)
def load_all():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        return None, None, None, None, 0, 0, "파일을 찾을 수 없습니다."
    try:
        model = CatBoostClassifier().load_model(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        
        # --- [수정 포인트: 모델이 요구하는 24개 변수 계산] ---
        # 1. 모델이 학습한 피처 리스트 (top_24_features)
        features = model.feature_names_

        # 2. '차이의 차이(diff)' 변수 계산 (데이터에 이미 deviation이 있다고 하셨으니 바로 뺍니다)
        df['viscosity_deviation_diff_3_2'] = df.get('stage3_viscosity_deviation', 0) - df.get('stage2_viscosity_deviation', 0)
        df['flow_deviation_diff_5_4'] = df.get('stage5_flow_deviation', 0) - df.get('stage4_flow_deviation', 0)
        df['viscosity_deviation_diff_5_1'] = df.get('stage5_viscosity_deviation', 0) - df.get('stage1_viscosity_deviation', 0)
        df['co2_deviation_diff_5_1'] = df.get('stage5_co2_deviation', 0) - df.get('stage1_co2_deviation', 0)

        # 3. '기울기(slope)' 변수 계산 (N2 편차 5개의 추세)
        n_cols = [f'stage{i}_n_deviation' for i in range(1, 6)]
        if all(c in df.columns for c in n_cols):
            # 행별로 S1~S5의 기울기를 구함
            df['n_deviation_slope'] = np.polyfit(range(5), df[n_cols].values.T, 1)[0]
        else:
            df['n_deviation_slope'] = 0

        # 4. 모델이 학습할 때 썼던 '순서 그대로' 24개 컬럼만 추출
        # 혹시라도 계산 안 된 컬럼이 있다면 0으로 채워 에러 방지
        for col in features:
            if col not in df.columns:
                df[col] = 0
        
        X_final = df[features].copy()
        # --------------------------------------------------

        # 예측 및 SHAP 계산 (기존 로직 유지)
        df['prob'] = (model.predict_proba(X_final)[:, 1] * 100).round(1)
        latest_pool = Pool(X_final.tail(1))
        latest_shap = model.get_feature_importance(data=latest_pool, type="ShapValues")[0, :-1]
        h_score = get_h_score(df, df.iloc[-1])
        
        return df, X_final, model, features, latest_shap, h_score, None
    except Exception as e:
        return None, None, None, None, None, 0, str(e)

df, X, model, feature_names, shap_vals, h_score, error_msg = load_all()
if error_msg:
    st.error(error_msg)
    st.stop()

# --- 증감 계산 로직 ---
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

temp_cols = [f'stage{i}_temp' for i in range(1, 6)]
hum_cols = [f'stage{i}_humidity' for i in range(1, 6)]

# 1. 불량률 증감 (전체 평균 비교)
curr_fail_rate = df['label'].mean() * 100
prev_fail_rate = df.iloc[:-1]['label'].mean() * 100 if len(df) > 1 else curr_fail_rate
diff_fail = curr_fail_rate - prev_fail_rate

# 2. 온도/습도 증감 (마지막 두 시점 비교)
diff_temp = latest[temp_cols].mean() - prev[temp_cols].mean()
diff_hum = latest[hum_cols].mean() - prev[hum_cols].mean()

def get_delta_html(diff):
    if diff > 0.01:
        return f'<span class="kpi-delta delta-up">▲ {abs(diff):.2f}</span>'
    elif diff < -0.01:
        return f'<span class="kpi-delta delta-down">▼ {abs(diff):.2f}</span>'
    else:
        return '<span class="kpi-delta delta-zero">─ 0.00</span>'

# 상단 UI 레이아웃
fail_prob = latest['prob']
p_color = "#a43434" if fail_prob >= 80 else "#f39c12" if fail_prob >= 51.48 else "#0d6e03"

def draw_traffic_light(score):
    r_cls = "tl-red" if score <= 70 else ""
    y_cls = "tl-yellow" if 70 < score <= 85 else ""
    g_cls = "tl-green" if score > 85 else ""
    return f'<div class="tl-container"><div class="tl-dot {r_cls}"></div><div class="tl-dot {y_cls}"></div><div class="tl-dot {g_cls}"></div></div>'

top_left, top_right = st.columns([1, 2])

with top_left:
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", 
        value=fail_prob,
        number={
            'suffix': "%", 
            'font': {'size': 35,'weight': 'bold'}, 
            'valueformat': '.1f'  # 🚀 80.0% 소수점 한자리 유지
        },
        title={'text': "불량 예측 확률", 'font': {'size': 18, 'weight': 'bold'}},
        gauge={
            'axis': {
                'range': [0, 100], 
                'tickwidth': 0,      # 🚀 눈금 실선 제거
                'showticklabels': True
            },
            'bar': {'color': p_color, 'thickness': 0.65},
            'bgcolor': "white",
            'borderwidth': 0,        # 🚀 외곽 테두리 제거
            'steps': [
                {'range': [0, 51.48], 'color': "#CBEAC4"},
                {'range': [51.48, 80], 'color': "#ffe8cb"},
                {'range': [80, 100], 'color': "#f5d2d2"}
            ]
        }
    ))
    
    # 3. 레이아웃 정리 (여백 및 배경)
    fig_gauge.update_layout(
        height=240, 
        margin=dict(t=50, b=10, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#404953"}
    )
    
    # 4. 출력
    st.plotly_chart(fig_gauge, use_container_width=True)

with top_right:
    st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)
    st.write("")
    if fail_prob >= 80:
        st.error(f"### 🚨 위험: 불량 발생 위험 매우 높음 ")
    elif fail_prob >= 51.48:
        st.warning(f"### ⚠️ 주의: 불량 패턴 감지 및 위험 고조")
    else:
        st.success(f"### ✅ 정상: 품질 안정 상태 유지")

    st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    
    # KPI 카드 렌더링
    k1.markdown(f"""<div class="kpi-card"><div class="kpi-label">설비건강(점)</div><div class="kpi-value">{h_score:.0f}</div><div class="kpi-bottom">{draw_traffic_light(h_score)}</div></div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="kpi-card"><div class="kpi-label">총 생산(개)</div><div class="kpi-value">{len(df):,}</div><div class="kpi-bottom"></div></div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="kpi-card"><div class="kpi-label">불량률(%)</div><div class="kpi-value">{curr_fail_rate:.1f}</div><div class="kpi-bottom">{get_delta_html(diff_fail)}</div></div>""", unsafe_allow_html=True)
    k4.markdown(f"""<div class="kpi-card"><div class="kpi-label">평균 온도(°C)</div><div class="kpi-value">{latest[temp_cols].mean():.1f}</div><div class="kpi-bottom">{get_delta_html(diff_temp)}</div></div>""", unsafe_allow_html=True)
    k5.markdown(f"""<div class="kpi-card"><div class="kpi-label">평균 습도(%)</div><div class="kpi-value">{latest[hum_cols].mean():.1f}</div><div class="kpi-bottom">{get_delta_html(diff_hum)}</div></div>""", unsafe_allow_html=True)

st.divider()

bot_left, bot_right = st.columns([1.8, 1.2])

with bot_left:
    st.subheader(" 실시간 불량 예측 확률 ")
    display_df = df.tail(30).copy()
    probs = display_df["prob"].to_numpy()
    n = len(probs)

    def rgba_by_prob(p, alpha):
        if p >= 80: return f"rgba(255,75,75,{alpha})"
        elif p >= 51.48: return f"rgba(255,170,0,{alpha})"
        else: return f"rgba(0,204,102,{alpha})"

    marker_colors = [rgba_by_prob(float(p), 1.0 if i == n - 1 else 0.4) for i, p in enumerate(probs)]

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=display_df['id'], y=display_df['prob'], mode='lines+markers',
        line=dict(color="#6e6e6e"), marker=dict(size=10, color=marker_colors),
        customdata=display_df['id'], hovertemplate="id=%{customdata}<br>prob=%{y:.2f}%<extra></extra>",
    ))
    fig_line.add_hline(y=51.48, line_dash="dot", line_color="orange")
    fig_line.add_hline(y=80, line_dash="dash", line_color="red")
    fig_line.update_layout(clickmode="event+select", height=280, margin=dict(t=10, b=10, l=10, r=10), yaxis_range=[0, 105], xaxis=dict(tickangle=-45,tickfont=dict(size=10)))

    event = st.plotly_chart(fig_line, use_container_width=True, on_select="rerun")
    
    if event and "selection" in event and event.get("selection", {}).get("points"):
        selected_id = event["selection"]["points"][0].get("customdata")
        if selected_id is not None:
            st.session_state["target_id"] = selected_id
            st.switch_page("pages/1_제품 상세 분석.py")

with bot_right:
    st.subheader(" 주요 영향 변수 ")
    
    # 1. SHAP 데이터를 데이터프레임으로 생성
    shap_df = pd.DataFrame({'eng_name': feature_names, 'val': shap_vals})
    
    # 2. [수정 포인트] 모든 파생변수를 포함하는 번역 로직
    def translate_feature(col):
        # (1) 특수 파생변수 강제 매핑 (가장 먼저 확인)
        manual_map = {
            'viscosity_deviation_diff_3_2': 'S3-S2 점도편차 차이',
            'flow_deviation_diff_5_4': 'S5-S4 유량편차 차이',
            'viscosity_deviation_diff_5_1': 'S5-S1 점도편차 차이',
            'co2_deviation_diff_5_1': 'S5-S1 CO2편차 차이',
            'n_deviation_slope': 'N2 편차 기울기',
            'temp_deviation': '전체 온도 편차',
            'hum_deviation': '전체 습도 편차'
        }
        if col in manual_map:
            return manual_map[col]
            
        # (2) 일반 스테이지 변수 처리 (stage1_flow_deviation 등)
        base_sensor_dict = {
            'temp': '온도', 'humidity': '습도', 'flow': '유량', 'density': '밀도',
            'viscosity': '점도', 'co2': 'CO2', 'o2': 'O2', 'n': 'N2'
        }
        
        if col.startswith('stage'):
            parts = col.split('_')
            stage_num = parts[0].replace('stage', 'S')  # stage1 -> S1
            sensor_eng = parts[1] if len(parts) > 1 else ""
            sensor_kor = base_sensor_dict.get(sensor_eng, sensor_eng)
            is_dev = ' 편차' if 'deviation' in col else ''
            return f"{stage_num} {sensor_kor}{is_dev}"
        
        # (3) 그 외 (언더바 제거)
        return col.replace('_', ' ')

    # 3. '항목' 컬럼에 한글 이름 주입
    shap_df['항목'] = shap_df['eng_name'].apply(translate_feature)
    
    # 4. 영향도가 큰 상위 5개 추출 (절대값 기준 정렬)
    shap_df['abs_val'] = shap_df['val'].abs()
    shap_df = shap_df.sort_values('abs_val', ascending=True).tail(5)

    colors = ['#67000d', '#a50f15', '#cb181d', '#ef3b2c', '#fb6a4a']
    
    # 5. 차트 그리기
    fig_shap = go.Figure(go.Bar(
        y=shap_df['항목'],      # ✅ 이제 여기서 한글 변수명 사용
        x=shap_df['val'], 
        orientation='h', 
        marker_color=[colors[i] if val > 0 else "#3f9adc" for i, val in enumerate(shap_df['val'])]
    ))
    
    fig_shap.update_layout(
        height=280, 
        margin=dict(t=0, b=10, l=10, r=50),
        yaxis=dict(tickfont=dict(size=12))
    )
    st.plotly_chart(fig_shap, use_container_width=True)
