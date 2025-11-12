import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------------------------------------
# PAGE CONFIGURATION
# -------------------------------------
st.set_page_config(page_title="JRU SHS Enrollment Forecast", page_icon="🎓", layout="centered")

st.title("🎓 Jose Rizal University SHS Enrollment Forecasting App")
st.write("""
This application leverages **Machine Learning (Decision Tree Regressor)** to predict 
future senior high school enrollments by strand and year level.
""")

# -------------------------------------
# LOAD MODEL
# -------------------------------------
@st.cache_resource
def load_model():
    """Load the trained DecisionTreeRegressor into a pipeline safely."""
    # Load the trained tree
    tree = joblib.load("JRU_SHS_DecisionTreeRegressor.joblib")

    # Preprocessor for categorical features
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ["Strand", "YearLevel"])],
        remainder="passthrough"
    )

    # Build pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", tree)
    ])

    # Fit preprocessor on dummy data to generate correct feature structure
    dummy_df = pd.DataFrame({
        "Strand": ["STEM", "ABM", "HUMSS", "TVL"],
        "YearLevel": ["Grade 11", "Grade 12", "Grade 11", "Grade 12"]
    })
    pipeline.named_steps["preprocessor"].fit(dummy_df)

    return pipeline

model = load_model()

# -------------------------------------
# USER INPUT
# -------------------------------------
st.subheader("🧮 Input Enrollment Details")
year_level = st.selectbox("Select Year Level:", ["Grade 11", "Grade 12"])
strand = st.selectbox("Select Strand:", ["STEM", "ABM", "HUMSS", "TVL"])
gender_ratio = st.slider("Estimated Male Student Percentage:", 0, 100, 50, step=5)

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

        male_pred = prediction * (gender_ratio / 100)
        female_pred = prediction - male_pred

        st.success(f"📊 **Predicted Total Enrollment:** {int(prediction)} students")
        st.write(f"👦 **Estimated Male Students:** {int(male_pred)}")
        st.write(f"👧 **Estimated Female Students:** {int(female_pred)}")

        # Plot bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Total", "Male", "Female"], [prediction, male_pred, female_pred],
               color=["#4B9CD3", "#87CEFA", "#FFB6C1"])
        plt.title(f"Predicted Enrollment Breakdown for {strand} ({year_level})")
        plt.ylabel("Number of Students")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# -------------------------------------
# DECISION TREE VISUALIZATION
# -------------------------------------
st.divider()
with st.expander("🌳 Show Decision Tree Visualization"):
    st.write("This diagram represents how the model splits data to make predictions.")

    tree_model = model.named_steps["regressor"]

    # Avoid feature_names to prevent IndexError
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(tree_model, filled=True, rounded=True, fontsize=10, ax=ax)
    st.pyplot(fig)