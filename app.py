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
Predict future senior high school enrollments by strand and year level.
This helps the university plan classrooms, faculty, and resources efficiently.
""")

# -------------------------------------
# LOAD PIPELINE
# -------------------------------------
@st.cache_resource
def load_model():
    """Load the full pipeline (preprocessor + DecisionTreeRegressor)"""
    pipeline = joblib.load("JRU_SHS_DecisionTree_FullPipeline.joblib")
    return pipeline

model = load_model()

# -------------------------------------
# USER INPUT
# -------------------------------------
st.subheader("🧮 Input Details")
strand = st.selectbox("Select Strand:", ["STEM", "ABM", "HUMSS", "TVL"])
year_level = st.selectbox("Select Year Level:", ["Grade 11", "Grade 12"])
projection_years = st.slider("Select number of years to project:", 1, 5, 3, step=1)

gender_split = st.checkbox("Show male/female split?", value=True)

# -------------------------------------
# PREDICTION
# -------------------------------------
if st.button("🔮 Predict Enrollment"):
    try:
        # Create projection dataframe
        current_year = 2025  # starting year
        years = [current_year + i for i in range(projection_years)]
        predictions = []

        for y in years:
            input_df = pd.DataFrame({"YearLevel": [year_level], "Strand": [strand]})
            pred = model.predict(input_df)[0]
            predictions.append(pred)

        # Display table
        df_proj = pd.DataFrame({
            "Year": years,
            "Predicted Enrollment": predictions
        })
        st.subheader("📊 Projected Enrollment")
        st.table(df_proj)

        # Plot line chart
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(years, predictions, marker='o', linestyle='-', color="#4B9CD3", label="Total")
        ax.set_title(f"Projected Enrollment for {strand} ({year_level})")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Students")
        ax.set_xticks(years)
        ax.grid(True)

        if gender_split:
            male = [p * 0.5 for p in predictions]  # assume 50/50 split if not known
            female = [p - m for p, m in zip(predictions, male)]
            ax.fill_between(years, 0, male, color="#87CEFA", alpha=0.6, label="Male")
            ax.fill_between(years, male, predictions, color="#FFB6C1", alpha=0.6, label="Female")
            ax.legend()

        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# -------------------------------------
# DECISION TREE VISUALIZATION
# -------------------------------------
st.divider()
with st.expander("🌳 Show Decision Tree Visualization"):
    st.write("This diagram shows how the model splits features to make predictions.")
    tree_model = model.named_steps["regressor"]

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(tree_model, filled=True, rounded=True, fontsize=10, ax=ax)
    st.pyplot(fig)
