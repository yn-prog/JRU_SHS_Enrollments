import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

# -------------------------------------
# PAGE CONFIGURATION
# -------------------------------------
st.set_page_config(page_title="JRU SHS Enrollment Dashboard", page_icon="🎓", layout="wide")

st.title("🎓 JRU SHS Enrollment Forecast & Dashboard")
st.write("""
This dashboard allows prediction of future enrollment and visualization of past enrollment data.
Upload historical enrollment CSV to explore trends and distributions.
""")

# -------------------------------------
# LOAD PIPELINE
# -------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("JRU_SHS_DecisionTree_FullPipeline.joblib")

model = load_model()

# -------------------------------------
# CSV UPLOAD
# -------------------------------------
st.sidebar.subheader("Upload Historical Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

historical_df = None
if uploaded_file:
    historical_df = pd.read_csv(uploaded_file, parse_dates=["DateEnrolled", "Birthdate"])
    st.sidebar.success("✅ File uploaded successfully!")

# -------------------------------------
# USER INPUT
# -------------------------------------
st.subheader("🧮 Predict Future Enrollment")
strand = st.selectbox("Select Strand:", ["STEM", "ABM", "HUMSS", "TVL"])
year_level = st.selectbox("Select Year Level:", ["Grade 11", "Grade 12"])
projection_years = st.slider("Number of years to project:", 1, 5, 3)

# -------------------------------------
# PREDICTION
# -------------------------------------
if st.button("🔮 Predict Enrollment"):
    current_year = 2025
    years = [current_year + i for i in range(projection_years)]
    predictions = []

    for y in years:
        input_df = pd.DataFrame({"YearLevel": [year_level], "Strand": [strand]})
        pred = model.predict(input_df)[0]
        predictions.append(pred)

    df_proj = pd.DataFrame({"Year": years, "Predicted Enrollment": predictions})
    st.subheader("📊 Projected Enrollment")
    st.table(df_proj)

    # Line chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(years, predictions, marker='o', color="#4B9CD3")
    ax.set_title(f"Projected Enrollment for {strand} ({year_level})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Students")
    ax.grid(True)
    st.pyplot(fig)

# -------------------------------------
# HISTORICAL DATA VISUALIZATION
# -------------------------------------
if historical_df is not None:
    st.subheader("📈 Historical Enrollment Data")

    # Total enrollment by year
    historical_df['Year'] = historical_df['DateEnrolled'].dt.year
    enroll_by_year = historical_df.groupby('Year').size().reset_index(name='Enrollment')
    st.line_chart(enroll_by_year.set_index('Year'))

    # Strand distribution
    st.subheader("Strand Distribution")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(data=historical_df, x='Strand', palette="pastel", order=historical_df['Strand'].value_counts().index)
    ax.set_title("Enrollment by Strand")
    st.pyplot(fig)

    # Gender distribution
    st.subheader("Gender Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    historical_df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=["#87CEFA","#FFB6C1"])
    plt.ylabel("")
    st.pyplot(fig)

    # Optional: Year vs Strand stacked bar chart
    st.subheader("Year vs Strand Enrollment")
    year_strand = historical_df.groupby(['Year', 'Strand']).size().unstack(fill_value=0)
    st.bar_chart(year_strand)
