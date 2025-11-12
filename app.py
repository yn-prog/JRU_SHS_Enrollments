import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# -------------------------------------
# PAGE CONFIG
# -------------------------------------
st.set_page_config(page_title="JRU SHS Enrollment Forecast", page_icon="🎓", layout="centered")

st.title("🎓 JRU SHS Enrollment Forecasting App")
st.write("""
Predict future senior high school enrollments at JRU by strand and year level.
This helps plan classrooms, teachers, and other resources efficiently.
""")

# -------------------------------------
# LOAD PIPELINE
# -------------------------------------
@st.cache_resource
def load_pipeline():
    """Load the trained full pipeline: preprocessor + DecisionTreeRegressor"""
    pipeline = joblib.load("JRU_SHS_DecisionTree_FullPipeline.joblib")
    return pipeline

model = load_pipeline()

# -------------------------------------
# USER INPUT
# -------------------------------------
st.subheader("🧮 Input Enrollment Details")
year_level = st.selectbox("Select Year Level:", ["Grade 11", "Grade 12"])
strand = st.selectbox("Select Strand:", ["STEM", "ABM", "HUMSS", "TVL"])
gender_ratio = st.slider("Estimated Male Student Percentage:", 0, 100, 50, step=5)

# Prepare input for the model
input_df = pd.DataFrame({"YearLevel": [year_level], "Strand": [strand]})

# -------------------------------------
# PREDICTION
# -------------------------------------
if st.button("🔮 Predict Enrollment"):
    try:
        # Predict total enrollment using the trained model
        total_students = model.predict(input_df)[0]

        # Calculate male/female split for visualization only
        male_students = total_students * (gender_ratio / 100)
        female_students = total_students - male_students

        st.success(f"📊 Predicted Total Enrollment: {int(total_students)} students")
        st.write(f"👦 Estimated Male Students: {int(male_students)}")
        st.write(f"👧 Estimated Female Students: {int(female_students)}")

        # Bar chart for visualization
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Total", "Male", "Female"], [total_students, male_students, female_students],
               color=["#4B9CD3", "#87CEFA", "#FFB6C1"])
        plt.title(f"Predicted Enrollment for {strand} ({year_level})")
        plt.ylabel("Number of Students")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# -------------------------------------
# DECISION TREE VISUALIZATION
# -------------------------------------
st.divider()
with st.expander("🌳 Show Decision Tree Structure"):
    st.write("This shows how the model splits features to predict total enrollment.")
    tree_model = model.named_steps["regressor"]

    fig, ax = plt.subplots(figsize=(20, 10))
    # Omit feature_names to prevent IndexError if preprocessing changes number of features
    plot_tree(tree_model, filled=True, rounded=True, fontsize=10, ax=ax)
    st.pyplot(fig)
