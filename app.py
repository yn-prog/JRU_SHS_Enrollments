import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# -------------------------------------
# PAGE CONFIGURATION
# -------------------------------------
st.set_page_config(page_title="JRU SHS Enrollment Forecast", page_icon="🎓", layout="centered")

st.title("🎓 Jose Rizal University SHS Enrollment Forecasting App")
st.write("""
This application leverages **Machine Learning (Decision Tree Regressor)** to predict 
future senior high school enrollments by strand and year level.  
It supports **Sustainable Development Goal 4 (Quality Education)** by helping administrators 
plan classrooms, resources, and faculty assignments efficiently.
""")

# -------------------------------------
# LOAD MODEL
# -------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("JRU_SHS_DecisionTree_Model.joblib")  # your saved model filename

model = load_model()

# -------------------------------------
# USER INPUT SECTION
# -------------------------------------
st.subheader("🧮 Input Enrollment Details")

year_level = st.selectbox("Select Year Level:", ["Grade 11", "Grade 12"])
strand = st.selectbox("Select Strand:", ["STEM", "ABM", "HUMSS", "TVL"])
gender_ratio = st.slider("Estimated Male Student Percentage:", 0, 100, 50, step=5)

# Convert input into dataframe for model
input_df = pd.DataFrame({
    "YearLevel": [year_level],
    "Strand": [strand]
})

# -------------------------------------
# PREDICTION
# -------------------------------------
if st.button("🔮 Predict Enrollment"):
    try:
        prediction = model.predict(input_df)[0]

        # Adjust prediction based on gender ratio (optional realism)
        male_pred = prediction * (gender_ratio / 100)
        female_pred = prediction - male_pred

        st.success(f"📊 **Predicted Total Enrollment:** {int(prediction)} students")
        st.write(f"👦 **Estimated Male Students:** {int(male_pred)}")
        st.write(f"👧 **Estimated Female Students:** {int(female_pred)}")

        # Bar chart visualization
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Total", "Male", "Female"], [prediction, male_pred, female_pred], color=["#4B9CD3", "#87CEFA", "#FFB6C1"])
        plt.title(f"Predicted Enrollment Breakdown for {strand} ({year_level})")
        plt.ylabel("Number of Students")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# -------------------------------------
# VISUALIZE DECISION TREE STRUCTURE
# -------------------------------------
st.divider()
with st.expander("🌳 Show Decision Tree Visualization"):
    st.write("This diagram represents how the model splits data to make predictions.")
    tree_model = model.named_steps['regressor']
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(tree_model, filled=True, feature_names=feature_names, rounded=True, fontsize=10)
    st.pyplot(fig)

# -------------------------------------
# FOOTER
# -------------------------------------
st.divider()
st.caption("Developed by JRU SHS Researchers | Powered by Python + Streamlit | © 2025")
