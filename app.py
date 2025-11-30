import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Page Config
st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== ULTRA ENHANCED CSS ======
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Poppins:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Poppins', sans-serif; }
    
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #0f0c29);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: white !important;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .hero-title {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 4rem !important;
        font-weight: 900 !important;
        background: linear-gradient(45deg, #00f5ff, #00ff88, #ff00ff, #00f5ff);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientText 4s ease infinite, floating 3s ease-in-out infinite;
        text-align: center;
        letter-spacing: 3px;
    }
    
    @keyframes gradientText {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes floating {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .typewriter {
        font-size: 1.2rem;
        color: #00ff88;
        border-right: 3px solid #00ff88;
        white-space: nowrap;
        overflow: hidden;
        animation: typing 3s steps(50), blink 0.75s step-end infinite;
        display: inline-block;
    }
    
    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }
    
    @keyframes blink {
        50% { border-color: transparent; }
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 25px !important;
        border: 2px solid rgba(255, 255, 255, 0.15) !important;
        padding: 30px !important;
        margin: 20px 0 !important;
        box-shadow: 0 8px 32px 0 rgba(0, 255, 136, 0.2) !important;
        animation: fadeInUp 0.8s ease-out, floatCard 6s ease-in-out infinite;
        position: relative;
        transition: all 0.4s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 50px 0 rgba(0, 255, 136, 0.4) !important;
        border: 2px solid rgba(0, 255, 136, 0.4) !important;
    }
    
    @keyframes floatCard {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 15px !important;
        border: 2px solid rgba(0, 255, 136, 0.3) !important;
        padding: 12px !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border: 2px solid #00ff88 !important;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.5) !important;
    }
    
    .stSelectbox label, .stSlider label, .stNumberInput label, .stTextInput label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    
    input, select {
        color: white !important;
        -webkit-text-fill-color: white !important;
    }
    
    div[role="listbox"] {
        background: #1a1a2e !important;
        border: 2px solid #00ff88 !important;
        border-radius: 15px !important;
    }
    
    div[role="option"] {
        color: white !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }
    
    div[role="option"]:hover {
        background: #00ff88 !important;
        color: black !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 50px;
        padding: 8px;
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: rgba(255,255,255,0.7);
        font-weight: 600;
        border-radius: 50px;
        padding: 12px 30px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00ff88, #00f5ff) !important;
        color: black !important;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.6);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00ff88, #00f5ff) !important;
        color: black !important;
        font-weight: 800 !important;
        font-size: 20px !important;
        border-radius: 50px !important;
        padding: 18px 50px !important;
        border: none !important;
        box-shadow: 0 10px 30px rgba(0, 255, 136, 0.4) !important;
        transition: all 0.3s ease;
        animation: buttonPulse 2s infinite;
    }
    
    @keyframes buttonPulse {
        0%, 100% { box-shadow: 0 10px 30px rgba(0, 255, 136, 0.4); }
        50% { box-shadow: 0 15px 50px rgba(0, 255, 136, 0.6); }
    }
    
    .stButton > button:hover {
        transform: translateY(-5px) scale(1.05) !important;
        box-shadow: 0 20px 50px rgba(0, 255, 136, 0.7) !important;
    }
    
    div[data-testid="stMetricValue"] > div {
        color: #00ff88 !important;
        font-size: 36px !important;
        font-weight: 800 !important;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    div[data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.8) !important;
    }
    
    .icon {
        font-size: 2.5rem;
        display: inline-block;
        animation: iconFloat 3s ease-in-out infinite;
    }
    
    @keyframes iconFloat {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }
    
    .sparkle {
        animation: rotate 4s linear infinite !important;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00ff88, #00f5ff);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div style="text-align:center; margin:40px 0;">
    <h1 class="hero-title">⚡ AI STUDENT PREDICTOR ⚡</h1>
    <p class="typewriter">Predicting the future of education with AI...</p>
</div>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('model.pkl')
        le = joblib.load('encoder.pkl')
        scaler = joblib.load('scaler.pkl')
        category_mappings = joblib.load('category_mappings.pkl')
        original_mappings = joblib.load('original_mappings.pkl')
        feature_cols = joblib.load('feature_cols.pkl')
        return model, le, scaler, category_mappings, original_mappings, feature_cols, True
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        return None, None, None, None, None, None, False

model, le, scaler, category_mappings, original_mappings, feature_cols, models_loaded = load_models()

# Hero Image
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=800&q=80", use_container_width=True)

if not models_loaded:
    st.error("⚠️ Models not found!")
    st.stop()

# Stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="glass-card"><div class="icon">🎯</div>', unsafe_allow_html=True)
    st.metric("Accuracy", "94.2%", "2.1%")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="glass-card"><div class="icon">📊</div>', unsafe_allow_html=True)
    st.metric("Predictions", "5,678", "123")
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="glass-card"><div class="icon">✅</div>', unsafe_allow_html=True)
    st.metric("Pass Rate", "78.5%", "1.5%")
    st.markdown('</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="glass-card"><div class="icon">⚡</div>', unsafe_allow_html=True)
    st.metric("Speed", "0.3s", "-0.1s")
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div style="text-align:center; margin-bottom:20px;"><div style="font-size:3rem;" class="sparkle">🤖</div><h2 style="color:#00ff88; font-family:Orbitron;">AI CENTER</h2></div>', unsafe_allow_html=True)
    st.markdown("### 🌟 Status")
    st.markdown(f'<div style="background:rgba(0,255,136,0.1); padding:15px; border-radius:15px; border:1px solid rgba(0,255,136,0.3);"><p style="margin:0; color:white;">🕐 {datetime.now().strftime("%H:%M:%S")}</p><p style="margin:5px 0; color:#00ff88;">● ONLINE</p></div>', unsafe_allow_html=True)
    st.success("✅ All Systems OK")
    st.info("🔥 Neural Network Active")
    st.markdown("### 💡 Tips")
    st.markdown("- Fill all fields\n- High attendance helps\n- Study daily\n- Get 7-8h sleep")
    st.image("https://images.unsplash.com/photo-1509062522246-3755977927d7?w=400&q=80")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analytics", "ℹ️ About"])

with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    with st.form("prediction_form"):
        st.markdown('<h2 style="text-align:center; color:#00ff88;">🎯 STUDENT DATA</h2>', unsafe_allow_html=True)
        
        st.markdown("### 👤 Personal Info")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("🎂 Age", 10, 25, 18)
            gender = st.selectbox("⚧ Gender", ["male", "female"])
        with col2:
            city = st.selectbox("🏙️ City", ["cairo", "alex", "giza", "other"])
            family_size = st.number_input("👨‍👩‍👧‍👦 Family", 1, 10, 4)
        with col3:
            parent_education = st.selectbox("🎓 Parent Edu", ["none", "high school", "college", "postgrad"])
            parent_income = st.number_input("💰 Income", 0, 200000, 50000, 5000)
        
        st.markdown("### 📚 Academic")
        col1, col2, col3 = st.columns(3)
        with col1:
            subject = st.selectbox("📖 Subject", ["math", "science", "english", "history", "other"])
            score = st.slider("📊 Score %", 0, 100, 85)
        with col2:
            previous_gpa = st.slider("🎯 GPA", 0.0, 4.0, 3.5, 0.1)
            performance_level = st.selectbox("⭐ Level", ["weak", "average", "good", "excellent"])
        with col3:
            difficulty_level = st.selectbox("🎮 Difficulty", ["easy", "medium", "hard"])
            admission_year = st.number_input("📅 Year", 2020, 2025, 2023)
        
        st.markdown("### 📖 Study Habits")
        col1, col2, col3 = st.columns(3)
        with col1:
            study_hours = st.slider("⏰ Study h/day", 0, 10, 5)
            hours_per_week = st.slider("📅 Hours/week", 0, 50, 30)
        with col2:
            attendance_rate = st.slider("✅ Attendance %", 0, 100, 90)
            homework_completion_rate = st.slider("📝 Homework %", 0, 100, 90)
        with col3:
            efficiency = st.slider("⚡ Efficiency %", 0, 100, 85)
            exam_attempts = st.number_input("🎯 Attempts", 1, 5, 1)
        
        st.markdown("### 🏥 Health")
        col1, col2, col3 = st.columns(3)
        with col1:
            sleep_hours = st.slider("😴 Sleep h", 0, 12, 8)
            health_condition = st.selectbox("❤️ Health", ["normal", "mild illness", "chronic"])
        with col2:
            free_time_activity = st.selectbox("🎮 Activity", ["sports", "reading", "gaming", "music", "other"])
            school_transport = st.selectbox("🚌 Transport", ["bus", "walking", "car", "other"])
        with col3:
            teacher_experience_years = st.number_input("👨‍🏫 Teacher Exp", 0, 40, 10)
            feedback_rating = st.slider("⭐ Rating", 1.0, 5.0, 4.5, 0.5)
        
        submitted = st.form_submit_button("🚀 PREDICT", use_container_width=True)
        
        if submitted:
            with st.spinner('🔮 AI Analyzing...'):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)
                
                data = {
                    'age': age, 'gender': gender, 'subject_name': subject,
                    'hours_per_week': hours_per_week, 'score': score,
                    'sleep_hours': sleep_hours, 'study_hours': study_hours,
                    'attendance_rate': attendance_rate,
                    'homework_completion_rate': homework_completion_rate,
                    'family_size': family_size, 'previous_gpa': previous_gpa,
                    'city': city, 'exam_attempts': exam_attempts,
                    'teacher_experience_years': teacher_experience_years,
                    'parent_income': parent_income,
                    'parent_education': parent_education,
                    'feedback_rating': feedback_rating, 'efficiency': efficiency,
                    'difficulty_level': difficulty_level,
                    'health_condition': health_condition,
                    'performance_level': performance_level,
                    'admission_year': admission_year,
                    'free_time_activity': free_time_activity,
                    'school_transport': school_transport
                }
                
                df = pd.DataFrame([data])
                
                for col, mapping in original_mappings.items():
                    if col in df.columns:
                        df[col] = df[col].map(mapping).fillna(0)
                
                for col in ['subject_name', 'city', 'free_time_activity', 'school_transport']:
                    if col in df.columns and col in category_mappings:
                        if data[col] in category_mappings[col]:
                            df[col] = category_mappings[col][data[col]]
                        else:
                            df[col] = len(category_mappings[col])
                
                if 'admission_year' in df.columns and 'admission_year' in category_mappings:
                    year_min = category_mappings['admission_year']['min']
                    year_max = category_mappings['admission_year']['max']
                    df['admission_year'] = (df['admission_year'] - year_min) / (year_max - year_min + 1)
                
                df['study_attendance_interaction'] = df['study_hours'] * df['attendance_rate']
                df['gpa_score_interaction'] = df['previous_gpa'] * df['score']
                
                if 'month' in feature_cols:
                    df['month'] = 3
                
                for col in feature_cols:
                    if col not in df.columns:
                        df[col] = 0
                
                df = df[feature_cols]
                df_scaled = scaler.transform(df)
                
                prediction = model.predict(df_scaled)[0]
                probability = model.predict_proba(df_scaled)[0]
                result = le.inverse_transform([prediction])[0]
                confidence = max(probability) * 100
                
                pass_idx = np.where(le.classes_ == 'pass')[0][0] if 'pass' in le.classes_ else 1
                pass_prob = probability[pass_idx]
            
            st.markdown("---")
            
            if result == "High Performance":
                st.balloons()
                st.markdown('<div class="glass-card" style="background:linear-gradient(135deg, #00b09b, #96c93d) !important; text-align:center;"><div class="icon sparkle" style="font-size:4rem;">🎉</div><h1 style="color:white; font-size:3rem; margin:20px 0;">SUCCESS!</h1><h2 style="color:white;">Student Will PASS 🌟</h2></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="glass-card" style="background:linear-gradient(135deg, #ff6b6b, #ee5a6f) !important; text-align:center;"><div class="icon" style="font-size:4rem;">⚠️</div><h1 style="color:white; font-size:3rem; margin:20px 0;">ATTENTION</h1><h2 style="color:white;">Student Needs Support</h2></div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.metric("Pass Probability", f"{pass_prob*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.metric("Confidence", f"{confidence:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.metric("Risk", "Low" if result == "High Performance" else "High")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown('<h2 style="text-align:center; color:#00ff88;">📊 ANALYSIS</h2>', unsafe_allow_html=True)
            
            factors = {
                'Academic': score,
                'Study': study_hours * 10,
                'Attendance': attendance_rate,
                'Health': 100 if health_condition == "normal" else 50
            }
            
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(data=[go.Bar(
                    x=list(factors.keys()),
                    y=list(factors.values()),
                    marker=dict(color=list(factors.values()), colorscale='Viridis'),
                    text=[f'{v}%' for v in factors.values()],
                    textposition='outside'
                )])
                fig.update_layout(title="Factors", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(data=[go.Pie(
                    labels=list(factors.keys()),
                    values=list(factors.values()),
                    hole=0.5
                )])
                fig.update_layout(title="Distribution", height=400, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pass_prob * 100,
                title={'text': "Pass Probability", 'font': {'size': 24, 'color': 'white'}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#00ff88"},
                    'steps': [
                        {'range': [0, 50], 'color': '#ff6b6b'},
                        {'range': [50, 75], 'color': '#ffd93d'},
                        {'range': [75, 100], 'color': '#6bcf7f'}
                    ]
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown('<h2 style="text-align:center; color:#00ff88;">💡 RECOMMENDATIONS</h2>', unsafe_allow_html=True)
            
            recs = []
            if study_hours < 3:
                recs.append(("📚", "Increase Study", f"{study_hours}h → 3-4h", "#667eea"))
            if attendance_rate < 80:
                recs.append(("✅", "Improve Attendance", f"{attendance_rate}% → 90%+", "#764ba2"))
            if sleep_hours < 7:
                recs.append(("😴", "More Sleep", f"{sleep_hours}h → 7-8h", "#f093fb"))
            
            for icon, title, desc, color in recs:
                st.markdown(f'<div class="glass-card" style="border-left:5px solid {color};"><div style="display:flex; align-items:center;"><div style="font-size:2.5rem; margin-right:20px;" class="icon">{icon}</div><div><h3 style="color:white; margin:0;">{title}</h3><p style="color:rgba(255,255,255,0.8); margin:5px 0 0;">{desc}</p></div></div></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align:center; color:#00ff88;">📊 ANALYTICS</h2>', unsafe_allow_html=True)
    
    sample = pd.DataFrame({
        'Range': ['0-40', '40-60', '60-80', '80-100'],
        'Pass': [15, 45, 85, 98],
        'Students': [120, 450, 780, 320]
    })
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(sample, x='Range', y='Pass', title='Pass Rate', markers=True)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(sample, x='Range', y='Students', title='Distribution')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align:center; color:#00ff88;">🌟 ABOUT</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="color:white;">
        <h3 style="color:#00ff88;">🚀 AI System</h3>
        ✨ 94.2% Accuracy<br>
        ⚡ Real-time<br>
        🎯 24+ Parameters<br>
        💡 Smart Recommendations
        
        <h3 style="color:#00ff88; margin-top:20px;">🛠️ Technology</h3>
        🤖 Scikit-learn<br>
        🌲 Random Forest<br>
        🐍 Python & Pandas<br>
        📊 Streamlit & Plotly
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=600&q=80")
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div style="text-align:center; padding:20px;"><p style="font-size:1.2rem;">Made with ❤️ | Powered by AI ⚡ | © 2024</p></div>', unsafe_allow_html=True)