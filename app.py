import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import base64
from pathlib import Path

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        min-width: 250px !important;
        max-width: 250px !important;
        display: block !important;
        visibility: visible !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)
# -----------------------------
# LOAD MODEL & DATA
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/student_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/StudentPerformanceFactors.csv")

model = load_model()
df = load_data()

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

.stApp{
background:#0f172a;
color:white;
}

section[data-testid="stSidebar"]{
background:#111827;
}

.block-container{
padding-top:1.2rem;
padding-bottom:2rem;
padding-left:2rem;
padding-right:2rem;
}

.hero{
background:linear-gradient(135deg,#2563eb,#1e3a8a);
padding:35px;
border-radius:18px;
text-align:center;
margin-bottom:25px;
box-shadow:0px 8px 20px rgba(0,0,0,.35);
}

.hero h1{
color:white;
font-size:42px;
font-weight:700;
margin-bottom:10px;
}

.hero p{
color:#e5e7eb;
font-size:18px;
}

.metric-card{
background:#1e293b;
padding:22px;
border-radius:15px;
text-align:center;
box-shadow:0px 5px 15px rgba(0,0,0,.25);
transition:0.3s;
}

.metric-card:hover{
transform:translateY(-4px);
}

.metric-title{
font-size:15px;
color:#cbd5e1;
}

.metric-value{
font-size:34px;
font-weight:bold;
color:#38bdf8;
}

.result-card{
padding:25px;
border-radius:18px;
font-size:28px;
font-weight:bold;
text-align:center;
color:white;
margin-top:20px;
box-shadow:0px 6px 20px rgba(0,0,0,.35);
}

.success-card{
background:#16a34a;
}

.good-card{
background:#2563eb;
}

.average-card{
background:#d97706;
}

.bad-card{
background:#dc2626;
}

.about-card{
background:#1e293b;
padding:25px;
border-radius:20px;
}

img{
border-radius:18px;
}

</style>
""",unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📚 Navigation")

page=st.sidebar.radio(
"Go To",
[
"🏠 Home",
"🎯 Prediction",
"📊 Dataset",
"📈 Visualizations",
"👨‍💻 About"
]
)
# =========================
# HOME PAGE
# =========================
if page == "🏠 Home":

    df = pd.read_csv("data/StudentPerformanceFactors.csv")

    st.markdown("""
    <div class="hero">
        <h1>🎓 Student Performance Prediction</h1>
        <p>
        Predict Student Exam Scores using Artificial Intelligence &
        Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("📊 Dataset Rows", f"{df.shape[0]}")

    with c2:
        st.metric("📈 Features", f"{df.shape[1]-1}")

    with c3:
        st.metric("🤖 Algorithm", "Linear Regression")

    with c4:
        st.metric("🎯 R² Score", "0.77")

    st.write("")

    st.subheader("📌 Project Overview")

    st.write("""
This End-to-End Machine Learning project predicts student exam scores using
academic, personal and environmental factors.

The project includes:

- Data Preprocessing
- Exploratory Data Analysis
- Machine Learning Pipeline
- Model Training
- Prediction Dashboard
- Interactive Visualizations
- Streamlit Deployment
""")

    st.write("")

    st.subheader("🛠 Technologies")

    a, b, c = st.columns(3)

    with a:
        st.success("🐍 Python")
        st.success("🐼 Pandas")
        st.success("📊 NumPy")

    with b:
        st.success("🤖 Scikit-Learn")
        st.success("📈 Matplotlib")
        st.success("🎨 Seaborn")

    with c:
        st.success("🌐 Streamlit")
        st.success("💾 Joblib")
        st.success("🔧 Git & GitHub")

    st.info(
        "Built as an End-to-End Machine Learning Portfolio Project."
    )

# =========================
# PREDICTION PAGE
# =========================
elif page == "🎯 Prediction":

    st.title("🎯 Student Performance Prediction")

    st.write("Fill in the student details below.")

    col1, col2 = st.columns(2)

    with col1:

        hours = st.slider("Hours Studied",0,40,10)

        attendance = st.slider("Attendance (%)",0,100,80)

        sleep = st.slider("Sleep Hours",4,10,7)

        previous = st.slider("Previous Scores",0,100,70)

        motivation = st.selectbox(
            "Motivation Level",
            ["Low","Medium","High"]
        )

        internet = st.selectbox(
            "Internet Access",
            ["Yes","No"]
        )

        tutoring = st.slider(
            "Tutoring Sessions",
            0,
            10,
            2
        )

        physical = st.slider(
            "Physical Activity",
            0,
            7,
            3
        )

        learning = st.selectbox(
            "Learning Disabilities",
            ["No","Yes"]
        )

        parental = st.selectbox(
            "Parental Education",
            ["High School","College","Postgraduate"]
        )

    with col2:

        parental_inv = st.selectbox(
            "Parental Involvement",
            ["Low","Medium","High"]
        )

        access = st.selectbox(
            "Access to Resources",
            ["Low","Medium","High"]
        )

        extracurricular = st.selectbox(
            "Extracurricular Activities",
            ["Yes","No"]
        )

        family_income = st.selectbox(
            "Family Income",
            ["Low","Medium","High"]
        )

        teacher = st.selectbox(
            "Teacher Quality",
            ["Low","Medium","High"]
        )

        school = st.selectbox(
            "School Type",
            ["Public","Private"]
        )

        peers = st.selectbox(
            "Peer Influence",
            ["Positive","Neutral","Negative"]
        )

        distance = st.selectbox(
            "Distance from Home",
            ["Near","Moderate","Far"]
        )

        gender = st.selectbox(
            "Gender",
            ["Male","Female"]
        )

    st.write("")

    if st.button("🎯 Predict Exam Score", use_container_width=True):

        input_df = pd.DataFrame({

            "Hours_Studied":[hours],

            "Attendance":[attendance],

            "Parental_Involvement":[parental_inv],

            "Access_to_Resources":[access],

            "Extracurricular_Activities":[extracurricular],

            "Sleep_Hours":[sleep],

            "Previous_Scores":[previous],

            "Motivation_Level":[motivation],

            "Internet_Access":[internet],

            "Tutoring_Sessions":[tutoring],

            "Family_Income":[family_income],

            "Teacher_Quality":[teacher],

            "School_Type":[school],

            "Peer_Influence":[peers],

            "Physical_Activity":[physical],

            "Learning_Disabilities":[learning],

            "Parental_Education_Level":[parental],

            "Distance_from_Home":[distance],

            "Gender":[gender]

        })

        prediction = model.predict(input_df)[0]

        score = max(0, min(100, float(prediction)))

        st.balloons()

        if score >= 90:

            color = "#16a34a"
            level = "🏆 Excellent Performance"

        elif score >= 75:

            color = "#22c55e"
            level = "😊 Very Good Performance"

        elif score >= 60:

            color = "#f59e0b"
            level = "🙂 Average Performance"

        else:

            color = "#ef4444"
            level = "⚠ Needs Improvement"

        st.markdown(
            f"""
<div style="background:{color};
padding:25px;
border-radius:18px;
text-align:center;
color:white;
margin-top:15px;">

<h2>Predicted Exam Score</h2>

<h1>{score:.2f}/100</h1>

<h3>{level}</h3>

</div>
""",
            unsafe_allow_html=True
        )

        st.progress(int(score))
# ==========================
# DATASET PAGE
# ==========================
elif page == "📊 Dataset":

    st.markdown("<h1>📊 Student Dataset</h1>", unsafe_allow_html=True)

    df = pd.read_csv("data/StudentPerformanceFactors.csv")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("---")

    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")

    st.subheader("Data Types")

    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(info, use_container_width=True)


# ==========================
# VISUALIZATIONS PAGE
# ==========================
elif page == "📈 Visualizations":

    st.markdown("<h1>📈 Exploratory Data Analysis</h1>", unsafe_allow_html=True)

    st.info("Visualizations created during Exploratory Data Analysis.")

    col1, col2 = st.columns(2)

    with col1:

      st.image(
        "images/exam_score_distribution.png",
        caption="Exam Score Distribution",
        use_container_width=True
    )

      st.image(
        "images/hours_vs_score.png",
        caption="Hours Studied vs Exam Score",
        use_container_width=True
    )

      st.image(
        "images/sleep_hours.png",
        caption="Sleep Hours vs Exam Score",
        use_container_width=True
    )

      st.image(
        "images/gender_score.png",
        caption="Gender vs Exam Score",
        use_container_width=True
    )

    with col2:

      st.image(
        "images/correlation_heatmap.png",
        caption="Correlation Heatmap",
        use_container_width=True
    )

      st.image(
        "images/attendance_vs_score.png",
        caption="Attendance vs Exam Score",
        use_container_width=True
    )

      st.image(
        "images/previous_scores.png",
        caption="Previous Scores vs Exam Score",
        use_container_width=True
    )
# ==========================
# ABOUT PAGE
# ========================
elif page == "👨‍💻 About":
    st.markdown("<h1>👨‍💻 About the Developer</h1>", unsafe_allow_html=True)

    left, right = st.columns([1, 2])

    with left:
        with open("images/profile.jpg", "rb") as f:
            data = base64.b64encode(f.read()).decode()
        st.markdown(f"""
            <img src="data:image/jpeg;base64,{data}" 
            style="border-radius: 50%; width: 200px; height: 200px; 
            object-fit: cover; display: block; margin: auto;">
        """, unsafe_allow_html=True)

    with right:
        st.markdown("## **Jivesh Mishra**")
        st.write("🎓 MSc Artificial Intelligence Student")
        st.write("💻 Python Developer")
        st.write("🤖 Machine Learning Enthusiast")
        st.write("📊 Data Science Learner")
        st.write("🌐 Streamlit Developer")
        st.write("🚀 Passionate about building AI-powered web applications.")

    st.markdown("---")

    st.subheader("Technical Skills")

    a, b, c = st.columns(3)

    with a:
        st.success("Python")
        st.success("Pandas")
        st.success("NumPy")

    with b:
        st.success("Scikit-Learn")
        st.success("Machine Learning")
        st.success("Data Analysis")

    with c:
        st.success("Streamlit")
        st.success("Git")
        st.success("GitHub")

    st.markdown("---")

    st.subheader("Career Objective")

    st.info("""
To become an AI & Machine Learning Engineer by building practical,
real-world projects that solve meaningful problems while continuously
learning modern AI technologies.
""")

    st.markdown("---")

    st.subheader("Connect with Me")

    st.markdown("""
**GitHub:** [github.com/jiveshai-07](https://github.com/jiveshai-07)

**LinkedIn:** [linkedin.com/in/jivesh-mishra](https://linkedin.com/in/jivesh-mishra)
""")

    st.success("⭐ Thank you for visiting my project!")
