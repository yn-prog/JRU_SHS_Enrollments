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

all_strands = list(hardcoded_predictions.keys())

# ---------------------------------------------------------
# PREDICTION SECTION (TOP BOX)
# ---------------------------------------------------------
with st.container():
    st.subheader("🔮 Predict Next Year's Enrollment")

    strand = st.selectbox("Select Strand:", all_strands)

    if st.button("✨ Predict Next Year"):
        next_year = 2026
        predicted_value = hardcoded_predictions[strand]

        st.markdown(f"""
        <div style="padding: 15px; background-color: #F7F7F7; border-radius: 10px; border: 1px solid #DDD;">
            <h3>🔮 Prediction for {strand} in {next_year}: 
            <span style="color:#005BBB;">{predicted_value} students</span></h3>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# DIVIDER
# ---------------------------------------------------------
st.divider()

# ---------------------------------------------------------
# CSV UPLOAD SECTION (UNDER PREDICTION)
# ---------------------------------------------------------
with st.container():
    st.subheader("📂 Upload Historical Enrollment CSV")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    historical_df = None
    if uploaded_file:
        historical_df = pd.read_csv(uploaded_file, parse_dates=["DateEnrolled", "Birthdate"])
        st.success("📁 File loaded successfully!")

# ---------------------------------------------------------
# HISTORICAL VISUALIZATION (ONLY IF CSV EXISTS)
# ---------------------------------------------------------
if historical_df is not None:

    st.divider()
    st.subheader("📈 Historical Enrollment Dashboard")

    historical_df["Year"] = historical_df["DateEnrolled"].dt.year

    # ---------------------------------------------------------
    # Enrollment Count by Year
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
    plt.close(fig)

    # ---------------------------------------------------------
    # Gender distribution
    # ---------------------------------------------------------
    st.write("### 🚹🚺 Gender Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))
    historical_df["Gender"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%"
    )
    plt.ylabel("")
    st.pyplot(fig)
    plt.close(fig)

    # ---------------------------------------------------------
    # Year vs Strand stacked chart
    # ---------------------------------------------------------
    st.write("### 📊 Enrollment by Year & Strand")

    year_strand = historical_df.groupby(["Year", "Strand"]).size().unstack(fill_value=0)
    st.bar_chart(year_strand)
