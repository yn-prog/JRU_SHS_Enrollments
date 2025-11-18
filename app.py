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
This application predicts future enrollment based only on **Strand**, 
following the trained Decision Tree model.  
Upload a historical dataset to view past enrollment trends.
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
st.subheader("🔮 Predict Enrollment by Strand")

strand = st.selectbox("Select Strand:", all_strands)

projection_years = st.slider(
    "Years to forecast:", 
    min_value=1, 
    max_value=5, 
    value=3
)

if st.button("✨ Predict Enrollment"):

    current_year = 2025
    future_years = [current_year + i for i in range(projection_years)]

    # model expects ONLY the features used in training: Strand
    df_input = pd.DataFrame({"Strand": [strand]})

    # Each future year gets the same prediction (model is strand-based only)
    pred_value = model.predict(df_input)[0]
    predictions = [pred_value] * projection_years

    proj_df = pd.DataFrame({
        "Year": future_years,
        "Predicted Enrollment": predictions
    })

    st.write("### 📊 Projection Results")
    st.table(proj_df)

    # Line plot
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(proj_df["Year"], proj_df["Predicted Enrollment"], marker="o")
    ax.set_title(f"Projected Enrollment for {strand}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Predicted Students")
    ax.grid(True)
    st.pyplot(fig)

# ---------------------------------------------------------
# HISTORICAL VISUALIZATION
# ---------------------------------------------------------
if historical_df is not None:

    st.subheader("📈 Historical Enrollment Dashboard")

    # Add Year column
    historical_df["Year"] = historical_df["DateEnrolled"].dt.year

    # -------------------------
