import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# -------------------------------------
# PAGE CONFIGURATION
# -------------------------------------
st.set_page_config(page_title="JRU SHS Enrollment Forecast", page_icon="🎓", layout="centered")

st.title("🎓 JRU SHS Enrollment Forecasting App")
st.write("""
Predict past and future senior high school enrollments by strand and year level.
Upload historical data to visualize trends and make projections.
""")

# -------------------------------------
# LOAD PIPELINE
# -------------------------------------
@st.cache_resource
def load_model():
    pipeline = joblib.load("JRU_SHS_DecisionTree_FullPipeline.joblib")
    return pipeline

model = load_model()

# -------------------------------------
# CSV UPLOAD
# -------------------------------------
st.subheader("📁 Upload Historical Enrollment Data (Optional)")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

historical_df = None
if uploaded_file is not None:
    historical_df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    # Preprocess dates if available
    if "DateEnrolled" in historical_df.columns:
        historical_df["Year"] = pd.to_datetime(historical_df["DateEnrolled"]).dt.year

# -------------------------------------
# USER INPUT
# -------------------------------------
st.subheader("🧮 Input Details")
strand = st.selectbox("Select Strand:", ["STEM", "ABM", "HUMSS", "TVL"])
year_level = st.selectbox("Select Year Level:", ["Grade 11", "Grade 12"])
projection_years = st.slider("Select number of years to project:", 1, 5, 3, step=1)

gender_split = st.checkbox("Show male/female split?", value=True)

# -------------------------------------
# PREDICTION AND VISUALIZATION
# -------------------------------------
if st.button("🔮 Show Enrollment Trends"):
    try:
        # --- Historical visualization ---
        fig, ax = plt.subplots(figsize=(10, 5))
        years = []
        past_enrollments = []

        if historical_df is not None:
            filtered = historical_df[
                (historical_df["Strand"] == strand) &
                (historical_df["YearLevel"] == year_level)
            ]
            if "Year" in filtered.columns:
                historical_counts = filtered.groupby("Year").size().sort_index()
                years = historical_counts.index.tolist()
                past_enrollments = historical_counts.values.tolist()
                ax.plot(years, past_enrollments, marker='o', linestyle='-', color="#4B9CD3", label="Past")

        # --- Future projections ---
        current_year = 2025 if len(years) == 0 else max(years) + 1
        proj_years = [current_year + i for i in range(projection_years)]
        predictions = []

        for y in proj_years:
            input_df = pd.DataFrame({"YearLevel": [year_level], "Strand": [strand]})
            pred = model.predict(input_df)[0]
            predictions.append(pred)

        ax.plot(proj_years, predictions, marker='o', linestyle='--', color="#FF7F0E", label="Projected")
        ax.set_title(f"Enrollment Trends for {strand} ({year_level})")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Students")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Optional gender breakdown
        if gender_split:
            st.info("Assuming 50/50 male/female split for projection")
            df_proj = pd.DataFrame({
                "Year": proj_years,
                "Total": predictions,
                "Male": [p * 0.5 for p in predictions],
                "Female": [p * 0.5 for p in predictions]
            })
            st.subheader("Projected Enrollment Details")
            st.table(df_proj)

    except Exception as e:
        st.error(f"⚠️ Failed to generate trends: {e}")

# -------------------------------------
# DECISION TREE VISUALIZATION
# -------------------------------------
st.divider()
with st.expander("🌳 Show Decision Tree Visualization"):
    st.write("Decision Tree splits for the model predictions:")
    tree_model = model.named_steps["regressor"]
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(tree_model, filled=True, rounded=True, fontsize=10, ax=ax)
    st.pyplot(fig)
