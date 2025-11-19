import streamlit as st
import pandas as pd
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
using **hardcoded final results** (no ML model needed).
""")

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
# HARDCODED PREDICTIONS
# ---------------------------------------------------------
hardcoded_predictions = {
    "SHS-ABM": 73,
    "SHS-AD": 175,
    "SHS-AN": 213,
    "SHS-CHSS": 241,
    "SHS-FB": 285,
    "SHS-HSSGA": 393,
    "SHS-SP": 485,
    "SHS-STEM": 711,
    "SHS-TG": 922
}

all_strands = list(hardcoded_predictions.keys())

# ---------------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------------
st.subheader("🔮 Predict Next Year's Enrollment")

strand = st.selectbox("Select Strand:", all_strands)

if st.button("✨ Predict Next Year"):
    
    next_year = 2026
    predicted_value = hardcoded_predictions[strand]

    st.write(f"## 🔮 Prediction for {strand} in {next_year}: **{predicted_value} students**")

    # ---------------------------------------------------------
    # CURRENT ENROLLMENT FROM CSV
    # ---------------------------------------------------------
    if historical_df is not None:

        current_count = historical_df[historical_df["Strand"] == strand].shape[0]

        st.write(f"### 📍 Current Enrollment ({strand}): **{current_count} students**")

        # Comparison chart
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Current", "Predicted Next Year"], [current_count, predicted_value])
        ax.set_title(f"Current vs Next Year Prediction\n({strand})")
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
