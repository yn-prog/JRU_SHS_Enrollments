import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="JRU SHS Enrollment Dashboard",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 JRU SHS Enrollment Forecast & Dashboard")
st.write("""
This app predicts **NEXT YEAR'S enrollment** based on Strand  
and displays **current enrollment** from the uploaded dataset.
""")

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("JRU_SHS_DecisionTree_FullPipeline.joblib")

model = load_model()

# ---------------------------------------------------------
# SIDEBAR — UPLOAD CSV
# ---------------------------------------------------------
st.sidebar.header("📂 Upload Historical Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

historical_df = None
if uploaded_file:
    historical_df = pd.read_csv(uploaded_file, parse_dates=["DateEnrolled", "Birthdate"])
    st.sidebar.success("📁 File loaded successfully!")

# ---------------------------------------------------------
# LIST OF ALL STRANDS (based on your dataset)
# ---------------------------------------------------------
all_strands = [
    "SHS-ABM",
    "SHS-AD",
    "SHS-AN",
    "SHS-CHSS",
    "SHS-FB",
    "SHS-HSSGA",
    "SHS-SP",
    "SHS-STEM",
    "SHS-TG"
]

# ---------------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------------
st.subheader("🔮 Predict Next Year's Enrollment")

strand = st.selectbox("Select Strand:", all_strands)

# The model WAS trained with numeric YearLevel (1 or 2)
year_level = st.selectbox("Select Year Level:", [1, 2])

if st.button("✨ Predict Next Year"):
    
    next_year = 2026  # Dataset year is 2024-2025

    # ---------------------------------------------------------
    # CORRECT INPUT TO MATCH TRAINING DATA
    # ---------------------------------------------------------
    input_df = pd.DataFrame({
        "YearLevel": [year_level],   
        "Strand": [strand]
    })

    # Model prediction
    predicted_value = model.predict(input_df)[0]

    st.write(f"## 🔮 Prediction for {strand} (Year Level {year_level}) in {next_year}: **{predicted_value:.0f} students**")

    # ---------------------------------------------------------
    # CURRENT ENROLLMENT (from uploaded CSV)
    # ---------------------------------------------------------
    if historical_df is not None:

        current_count = historical_df[
            (historical_df["Strand"] == strand) &
            (historical_df["YearLevel"] == year_level)
        ].shape[0]

        st.write(f"### 📍 Current Enrollment ({strand}, YearLevel {year_level}): **{current_count} students**")

        # Comparison chart
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Current", "Predicted Next Year"], [current_count, predicted_value])
        ax.set_title(f"Current vs Next Year Prediction\n({strand}, YearLevel {year_level})")
        ax.set_ylabel("Number of Students")
        st.pyplot(fig)

# ---------------------------------------------------------
# HISTORICAL VISUALIZATION
# ---------------------------------------------------------
if historical_df is not None:

    st.subheader("📈 Historical Enrollment Dashboard")

    # Add year column based on DateEnrolled
    historical_df["Year"] = historical_df["DateEnrolled"].dt.year

    # ---------------------------------------------------------
    # Total enrollment by year
    # ---------------------------------------------------------
    st.write("### 🗓 Enrollment Count by Year")
    enroll_by_year = historical_df.groupby("Year").size().reset_index(name="Enrollment")
    st.line_chart(enroll_by_year.set_index("Year"))

    # ---------------------------------------------------------
    # Strand distribution
    # ---------------------------------------------------------
    st.write("### 🧭 Strand Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=historical_df,
        x="Strand",
        palette="pastel",
        order=historical_df["Strand"].value_counts().index
    )
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ---------------------------------------------------------
    # Gender distribution
    # ---------------------------------------------------------
    st.write("### 🚹🚺 Gender Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))
    historical_df["Gender"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        colors=["#87CEFA", "#FFB6C1"]
    )
    plt.ylabel("")
    st.pyplot(fig)

    # ---------------------------------------------------------
    # Year vs Strand stacked bar chart
    # ---------------------------------------------------------
    st.write("### 📊 Enrollment by Year & Strand")

    year_strand = historical_df.groupby(["Year", "Strand"]).size().unstack(fill_value=0)
    st.bar_chart(year_strand)
