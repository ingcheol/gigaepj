import streamlit as st
import pandas as pd
import joblib

# 모델 불러오기
model = joblib.load('xgb_model.pkl')

# 타이틀
st.title("🧬 고혈압 위험도 예측기")
st.markdown("건강 정보를 입력하면 고혈압 위험도를 예측합니다.")

# 사용자 입력 받기
sex = st.selectbox("성별", ["남자", "여자"])
age = st.slider("나이", 10, 100, 30)
bs3_2 = st.slider("하루 평균 궐련 흡연량 (개비)", 0, 60, 5)
bs12_47 = st.selectbox("전자담배 사용 여부", ["사용 안 함", "사용 중"])
bmi = st.slider("체질량지수(BMI)", 10.0, 50.0, 22.0)

# 입력값 변환
sex_val = 0 if sex == "남자" else 1
bs12_val = 1 if bs12_47 == "사용 중" else 0

# 예측용 데이터프레임 생성
user_input = pd.DataFrame([{
    "sex": sex_val,
    "age": age,
    "BS3_2": bs3_2,
    "BS12_47": bs12_val,
    "HE_BMI": bmi
}])

# 예측
if st.button("🔍 예측하기"):
    threshold = 0.7
    proba = model.predict_proba(user_input)[0][1]
    prediction = int(proba >= threshold)

    st.markdown(f"### 🧿 고혈압 위험도: **{proba * 100:.2f}%**")

    if prediction == 1:
        st.warning("⚠️ 고혈압 위험이 **매우 높습니다.** 전문가 상담을 권장합니다.")
    else:
        st.success("✅ 고혈압 위험이 낮습니다. 하지만 정기적인 건강 검진을 권장합니다.")
