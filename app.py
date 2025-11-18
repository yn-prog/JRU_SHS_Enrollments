import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="JRU SHS Enrollment Dashboard",
                   page_icon="🎓",
                   layout="wide")

st.title("🎓 JRU SHS Enrollment Forecast & Dashboard")
st.write("""
Upload historical enrollment data and generate predictions using the trained Decision Tree Model.
""")

# ---------------------------------------------------------
# LOAD MODEL (FULL PIPELINE)
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
# PREDICTION SECTION
# ---------------------------------------------------------
st.subheader("🔮 Predict Future Enrollment")

# Dynamic: extract strands from HISTORICAL DATA if uploaded
if historical_df is not None:
    strands = sorted(historical_df["Strand"].unique())
else:
    # Fallback if no CSV uploaded
    strands = [
        "SHS-ABM","SHS-AD","SHS-AN","SHS-CHSS",
        "SHS-FB","SHS-HSSGA","SHS-SP","SHS-STEM","SHS-TG"
    ]

strand = st.selectbox("Select Strand:", strands)
year_level = st.selectbox("Select Year Level:", ["Grade 11", "Grade 12"])
projection_years = st.slider("Years to forecast ahead:", 1, 5, 3)

if st.button("✨ Predict Enrollment"):
    current_year = 2025
    future_years = [current_year + i for i in range(projection_years)]
    predictions = []

    for yr in future_years:
        df_input = pd.DataFrame({
            "YearLevel": [year_level],
            "Strand": [strand]
        })
        pred = model.predict(df_input)[0]
        predictions.append(pred)

    proj_df = pd.DataFrame({
        "Year": future_years,
        "Predicted Enrollment": predictions
    })

    st.write("### 📊 Projection Results")
    st.table(proj_df)

    # Plot
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(proj_df["Year"], proj_df["Predicted Enrollment"], marker="o")
    ax.set_title(f"Projected Enrollment for {strand}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Enrollment")
    st.pyplot(fig)

# ---------------------------------------------------------
# HISTORICAL VISUALIZATION
# ---------------------------------------------------------
if historical_df is not None:

    st.subheader("📈 Historical Enrollment Dashboard")

    # Add Year column
    historical_df["Year"] = historical_df["DateEnrolled"].dt.year

    # -------------------------
    # Total Enrollment per Year
    # -------------------------
    st.write("### 🗓 Total Enrollment Over Time")
    yearly_counts = historical_df.groupby("Year").size().reset_index(name="Enrollment")
    st.line_chart(yearly_counts.set_index("Year"))

    # -------------------------
    # Strand Distribution
    # -------------------------
    st.write("### 🎒 Strand Distribution")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.countplot(data=historical_df,
                  x="Strand",
                  order=historical_df["Strand"].value_counts().index,
                  palette="pastel")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # -------------------------
    # Gender Pie Chart
    # -------------------------
    st.write("### ⚤ Gender Distribution")
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    historical_df["Gender"].value_counts().plot(
        kind="pie", autopct="%.1f%%"
    )
    ax2.set_ylabel("")
    st.pyplot(fig2)

    # -------------------------
    # Year vs Strand (stacked)
    # -------------------------
    st.write("### 🧱 Enrollment by Strand per Year")
    year_strand = historical_df.groupby(["Year", "Strand"]).size().unstack(fill_value=0)
    st.bar_chart(year_strand)
