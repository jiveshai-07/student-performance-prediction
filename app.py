import streamlit as st
import pandas as pd
import joblib

# Load custom CSS
def load_css():
    with open("styles/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# Page Configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📚 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🎯 Prediction",
        "📊 Dataset",
        "📈 Visualizations",
        "ℹ️ About"
    ]
)

# -------------------------------
# Home Page
# -------------------------------
if page == "🏠 Home":

    st.markdown("""
<div style='
background: linear-gradient(90deg,#0d1117,#161b22);
padding:30px;
border-radius:15px;
border:1px solid #30363d;
text-align:center;
margin-bottom:20px;'>

<h1 style='color:#58a6ff;margin-bottom:10px;'>
🎓 Student Performance Prediction
</h1>

<h4 style='color:white;'>
Predict Student Exam Scores using Artificial Intelligence & Machine Learning
</h4>

<p style='color:#c9d1d9;font-size:18px;'>
Built with Python • Scikit-Learn • Streamlit
</p>

</div>
""", unsafe_allow_html=True)

st.markdown("---")
df = pd.read_csv("data/StudentPerformanceFactors.csv")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Dataset Rows", df.shape[0])

    with col2:
        st.metric("📈 Features", df.shape[1])

    with col3:
        st.metric("🤖 Algorithm", "Linear Regression")

    with col4:
        st.metric("🎯 R² Score", "0.77")

    st.markdown("---")

    st.header("📌 Project Overview")

    st.write("""
This project predicts a student's exam score using Machine Learning.

The model analyzes various academic and personal factors to estimate the expected exam score.

The project demonstrates the complete Machine Learning workflow:

- Data Collection
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Pipeline
- Model Evaluation
- Web App Development
- Deployment using Streamlit
""")

    st.markdown("---")

    st.header("🛠 Technologies Used")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.success("🐍 Python")
        st.success("🐼 Pandas")
        st.success("🔢 NumPy")

    with tech2:
        st.success("📊 Matplotlib")
        st.success("🎨 Seaborn")
        st.success("🤖 Scikit-Learn")

    with tech3:
        st.success("🌐 Streamlit")
        st.success("💾 Joblib")
        st.success("📁 Git & GitHub")

    st.markdown("---")

    st.info(
        "💡 This project was built as part of my Machine Learning & Data Science portfolio."
    )

# -------------------------------
# Prediction Page
# -------------------------------
elif page == "🎯 Prediction":

    

    st.title("🎯 Student Performance Prediction")

    st.markdown(
        "Fill in the student's information below and click **Predict Exam Score**."
    )

    # Load trained pipeline
    model = joblib.load("models/student_model.pkl")

    col1, col2 = st.columns(2)

    with col1:

        hours = st.number_input(
            "Hours Studied",
            min_value=0,
            max_value=24,
            value=5
        )

        attendance = st.number_input(
            "Attendance (%)",
            min_value=0,
            max_value=100,
            value=80
        )

        previous = st.number_input(
            "Previous Scores",
            min_value=0,
            max_value=100,
            value=70
        )

        sleep = st.number_input(
            "Sleep Hours",
            min_value=0,
            max_value=12,
            value=7
        )

        tutoring = st.number_input(
            "Tutoring Sessions",
            min_value=0,
            max_value=20,
            value=2
        )

        physical = st.number_input(
            "Physical Activity (hrs/week)",
            min_value=0,
            max_value=20,
            value=5
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        school = st.selectbox(
            "School Type",
            ["Public", "Private"]
        )

        internet = st.selectbox(
            "Internet Access",
            ["Yes", "No"]
        )

    with col2:

        parental = st.selectbox(
            "Parental Involvement",
            ["Low", "Medium", "High"]
        )

        resources = st.selectbox(
            "Access to Resources",
            ["Low", "Medium", "High"]
        )

        extra = st.selectbox(
            "Extracurricular Activities",
            ["Yes", "No"]
        )

        motivation = st.selectbox(
            "Motivation Level",
            ["Low", "Medium", "High"]
        )

        income = st.selectbox(
            "Family Income",
            ["Low", "Medium", "High"]
        )

        teacher = st.selectbox(
            "Teacher Quality",
            ["Low", "Medium", "High"]
        )

        peer = st.selectbox(
            "Peer Influence",
            ["Positive", "Neutral", "Negative"]
        )

        disability = st.selectbox(
            "Learning Disabilities",
            ["Yes", "No"]
        )

        education = st.selectbox(
            "Parental Education Level",
            ["High School", "College", "Postgraduate"]
        )

        distance = st.selectbox(
            "Distance from Home",
            ["Near", "Moderate", "Far"]
        )

    st.markdown("---")

    if st.button("🎯 Predict Exam Score", use_container_width=True):

        input_data = pd.DataFrame({

            "Hours_Studied":[hours],
            "Attendance":[attendance],
            "Parental_Involvement":[parental],
            "Access_to_Resources":[resources],
            "Extracurricular_Activities":[extra],
            "Sleep_Hours":[sleep],
            "Previous_Scores":[previous],
            "Motivation_Level":[motivation],
            "Internet_Access":[internet],
            "Tutoring_Sessions":[tutoring],
            "Family_Income":[income],
            "Teacher_Quality":[teacher],
            "School_Type":[school],
            "Peer_Influence":[peer],
            "Physical_Activity":[physical],
            "Learning_Disabilities":[disability],
            "Parental_Education_Level":[education],
            "Distance_from_Home":[distance],
            "Gender":[gender]

        })

        prediction = model.predict(input_data)

        st.success(
            f"🎓 Predicted Exam Score : {prediction[0]:.2f}"
        )

        st.balloons()
    

# -------------------------------
# Dataset Page
# -------------------------------
elif page == "📊 Dataset":

    

    st.title("📊 Student Dataset")

    st.markdown("Explore the dataset used to train the Machine Learning model.")

    st.markdown("---")

    # Load dataset
    df = pd.read_csv("data/StudentPerformanceFactors.csv")

    # ==========================
    # Dataset Metrics
    # ==========================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📄 Rows", df.shape[0])

    with col2:
        st.metric("📑 Columns", df.shape[1])

    with col3:
        st.metric("❓ Missing Values", int(df.isnull().sum().sum()))

    st.markdown("---")

    # ==========================
    # Dataset Preview
    # ==========================

    st.subheader("👀 Dataset Preview")

    rows = st.slider(
        "Select number of rows",
        min_value=5,
        max_value=50,
        value=10
    )

    st.dataframe(df.head(rows), use_container_width=True)

    st.markdown("---")

    # ==========================
    # Dataset Information
    # ==========================

    st.subheader("📋 Dataset Information")

    info_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(info_df, use_container_width=True)

    st.markdown("---")

    # ==========================
    # Missing Values
    # ==========================

    st.subheader("❓ Missing Values")

    missing = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(missing, use_container_width=True)

    st.markdown("---")

    # ==========================
    # Statistical Summary
    # ==========================

    st.subheader("📊 Statistical Summary")

    st.dataframe(df.describe(), use_container_width=True)

# -------------------------------
# Visualizations
# -------------------------------
elif page == "📈 Visualizations":

    st.title("📈 Exploratory Data Analysis")

    st.markdown(
        "These visualizations were generated during the Exploratory Data Analysis (EDA) phase."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Exam Score Distribution")
        st.image(
            "images/exam_score_distribution.png",
            use_container_width=True
        )

        st.caption(
            "Shows how exam scores are distributed among students."
        )

        st.markdown("---")

        st.subheader("📚 Hours Studied vs Exam Score")
        st.image(
            "images/hours_vs_score.png",
            use_container_width=True
        )

        st.caption(
            "Students studying more hours generally achieve higher scores."
        )

        st.markdown("---")

        st.subheader("😴 Sleep Hours vs Exam Score")
        st.image(
            "images/sleep_hours.png",
            use_container_width=True
        )

        st.caption(
            "Relationship between sleep duration and academic performance."
        )

        st.markdown("---")

        st.subheader("👨👩 Gender vs Exam Score")
        st.image(
            "images/gender_score.png",
            use_container_width=True
        )

        st.caption(
            "Comparison of exam scores across genders."
        )

    with col2:

        st.subheader("🔥 Correlation Heatmap")
        st.image(
            "images/correlation_heatmap.png",
            use_container_width=True
        )

        st.caption(
            "Correlation between all numerical features."
        )

        st.markdown("---")

        st.subheader("🏫 Attendance vs Exam Score")
        st.image(
            "images/attendance_vs_score.png",
            use_container_width=True
        )

        st.caption(
            "Higher attendance is generally associated with better scores."
        )

        st.markdown("---")

        st.subheader("📝 Previous Scores vs Exam Score")
        st.image(
            "images/previous_scores.png",
            use_container_width=True
        )

        st.caption(
            "Students with stronger previous performance often score higher."
        )

    st.markdown("---")

    st.success("✅ All visualizations were created using Matplotlib and Seaborn.")

# -------------------------------
# About
# -------------------------------
elif page == "ℹ️ About":

    st.title("👨‍💻 About the Developer")

    st.markdown("---")

    col1, col2 = st.columns([1,2])

    with col1:

        st.image(
            "https://avatars.githubusercontent.com/u/9919?s=280&v=4",
            width=180
        )

    with col2:

        st.header("Jivesh Mishra")

        st.write("🎓 BCA Graduate")

        st.write("🤖 Currently pursuing M.Sc. in Artificial Intelligence")

        st.write("💡 Passionate about AI, Machine Learning and Data Science")

        st.write("🚀 Building real-world AI projects to strengthen my portfolio")

    st.markdown("---")

    st.header("💻 Technical Skills")

    skill1, skill2, skill3 = st.columns(3)

    with skill1:
        st.success("Python")
        st.success("Pandas")
        st.success("NumPy")
        st.success("SQL")

    with skill2:
        st.success("Machine Learning")
        st.success("Data Science")
        st.success("Scikit-Learn")
        st.success("EDA")

    with skill3:
        st.success("Git")
        st.success("GitHub")
        st.success("Streamlit")
        st.success("Data Visualization")

    st.markdown("---")

    st.header("🎯 Career Objective")

    st.info("""
I am passionate about solving real-world problems using Artificial Intelligence,
Machine Learning and Data Science.

My goal is to become an AI/ML Engineer by building impactful projects,
continuously learning new technologies, and applying them to practical solutions.
""")

    st.markdown("---")

    st.header("🔗 Connect with Me")

    st.markdown(
        """
**GitHub**

https://github.com/jiveshmishra

**LinkedIn**

https://www.linkedin.com/in/jivesh-mishra-01433331b
"""
    )

    st.markdown("---")

    st.success("⭐ Thank you for visiting my project!")