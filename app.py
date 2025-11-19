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
This app predicts **NEXT YEAR'S enrollment** based on Strand.
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
# PREDICTIONS
# ---------------------------------------------------------
hardcoded_predictions = {
    "SHS-AN": 73,
    "SHS-TG": 175,
    "SHS-SP": 213,
    "SHS-FB": 241,
    "SHS-CHSS": 285,
    "SHS-AD": 393,
    "SHS-ABM": 485,
    "SHS-HSSGA": 711,
    "SHS-STEM": 922
}

# ---------------------------------------------------------
# CURRENT ENROLLMENT VALUES
# ---------------------------------------------------------
hardcoded_current_enrollment = {
    "SHS-AN": 33,
    "SHS-TG": 44,
    "SHS-SP": 50,
    "SHS-FB": 83,
    "SHS-CHSS": 89,
    "SHS-AD": 138,
    "SHS-ABM": 293,
    "SHS-HSSGA": 332,
    "SHS-STEM": 922
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

    # Current enrollment
    if historical_df is not None:
        current_count = historical_df[historical_df["Strand"] == strand].shape[0]
    else:
        current_count = hardcoded_current_enrollment[strand]

    st.write(f"### 📍 Current Enrollment ({strand}): **{current_count} students**")

# ---------------------------------------------------------
# HISTORICAL VISUALIZATION
# ---------------------------------------------------------
if historical_df is not None:

    st.subheader("📈 Historical Enrollment Dashboard")

    historical_df["Year"] = historical_df["DateEnrolled"].dt.year

    st.write("### 🗓 Enrollment Count by Year")
    enroll_by_year = historical_df.groupby("Year").size().reset_index(name="Enrollment")
    st.line_chart(enroll_by_year.set_index("Year"))

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
    plt.close(fig)

    st.write("### 🚹🚺 Gender Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    historical_df["Gender"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%"
    )
    plt.ylabel("")
    st.pyplot(fig)
    plt.close(fig)

    st.write("### 📊 Enrollment by Year & Strand")
    year_strand = historical_df.groupby(["Year", "Strand"]).size().unstack(fill_value=0)
    st.bar_chart(year_strand)
