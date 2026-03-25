import joblib
import pandas as pd
import streamlit as st

from agent import graph

# Load resources
model_pipeline = joblib.load('admission_pipeline.pkl')
features = joblib.load('feature_names.pkl')

# --- STREAMLIT UI ---
st.set_page_config(page_title="FlexiSAF Admission AI", layout="centered")
st.title("🎓 Admission Success Predictor")
st.markdown("### AI-Powered Graduate School Guidance")

with st.sidebar:
    st.header("Enter Student Metrics")
    gre = st.number_input("GRE Score", 260, 340, 310)
    toefl = st.number_input("TOEFL Score", 0, 120, 100)
    rating = st.slider("University Rating", 1, 5, 3)
    sop = st.slider("SOP Strength", 1.0, 5.0, 3.5)
    lor = st.slider("LOR Strength", 1.0, 5.0, 3.5)
    cgpa = st.number_input("CGPA", 0.0, 10.0, 8.5)
    research = st.radio("Research Experience?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

if st.button("Analyze Application"):
    # Prepare Prediction Data 
    # Use underscores to match the cleaned training columns
    input_dict = {
        'GRE_Score': gre, 
        'TOEFL_Score': toefl, 
        'University_Rating': rating,
        'SOP': sop, 
        'LOR': lor, 
        'CGPA': cgpa, 
        'Research': research
    }
    
    # Create DataFrame and ensure column order matches training exactly
    input_df = pd.DataFrame([input_dict])[features]
    
    # Get Probability
    prob_array = model_pipeline.predict_proba(input_df)
    prob = prob_array[0][1]
    pred = 1 if prob >= 0.5 else 0

    # 2. GenAI Explanation via LangGraph
    with st.spinner("Consulting AI Counselor..."):
        try:
            result = graph.invoke({
                "input_data": input_dict, 
                "prediction": pred, 
                "probability": round(prob * 100, 2)
            })
            ai_text = result.get('explanation', "AI Counselor is currently unavailable.")
        except Exception as e:
            ai_text = f"Error connecting to Gemini: {e}"

    # UI Display
    st.divider()
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Success Probability", f"{round(prob * 100, 2)}%")
        if pred == 1:
            st.success("High Admission Probability!")
        else:
            st.warning("Low Admission Probability.")

    with col2:
        st.subheader("AI Insight")
        st.info(ai_text)

    # Feature Impact Analysis
    st.divider()
    st.subheader("What influenced your result?")
    importances = model_pipeline.named_steps['classifier'].feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
    st.bar_chart(feat_imp)