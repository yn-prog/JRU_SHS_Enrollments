import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="JRU SHS Enrollment Dashboard",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------------
# APP TITLE
# ---------------------------------------------------------
st.title("🎓 JRU SHS Enrollment Forecast & Dashboard")

# ---------------------------------------------------------
# DETAILED INTRODUCTION
# ---------------------------------------------------------
st.markdown("""
<div style="padding: 15px; background-color: #F7F7F7; border-radius: 10px; border: 1px solid #DDD;">

### 📘 About This Dashboard
The **JRU Senior High School Enrollment Forecasting Dashboard** is an interactive tool designed to support **evidence-based decision-making** at José Rizal University. This dashboard forms part of a research study focused on addressing persistent challenges in educational resource allocation—an issue that continues to affect many schools across the Philippines.

### 🎓 Context of the Study
Education remains one of the most crucial sectors within the Philippine government. However, the country continues to face longstanding challenges such as shortages of instructional materials, insufficient school infrastructure, limited facilities, and inadequate teaching staff. These barriers hinder the delivery of quality education and contribute to learning inequalities (Coloquit, 2020).

These issues relate directly to **Sustainable Development Goal (SDG 4 — Quality Education)**, which aims to ensure inclusive, equitable, and effective learning opportunities for all. Efficient resource planning, particularly in Senior High School, plays a critical role in achieving these goals.

### 🏫 Why Focus on Senior High School Enrollment?
Senior High School (SHS), the final stage of the K–12 program, prepares students for specialized academic or technical-vocational pathways. Each SHS strand—such as STEM, ABM, HUMSS, or TVL—requires different facilities, staffing, equipment, and budget allocations.

Because enrollment numbers fluctuate every year, the university faces difficulties such as:
- Overcrowded classrooms  
- Underutilized rooms in some strands  
- Budget misalignment  
- Facility shortages (e.g., laboratories, business rooms, TVL equipment)  
- Scheduling inefficiencies before classes begin  

Currently, administrators rely heavily on simple comparisons of past enrollment counts. This method is not enough to reflect trends, sudden increases, or strand-specific demands.

### 📊 Purpose of This Dashboard
This dashboard introduces a **data-driven forecasting system** that predicts the number of incoming Senior High School students per strand for the next academic year. Its goals are to:
- Provide **accurate, strand-level enrollment forecasts**
- Support **better classroom and facility planning**
- Improve **budget allocation** based on predicted strand needs
- Ensure **equitable access** to learning spaces and educational resources
- Assist school leaders in making **timely and well-informed decisions**

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------
predictions = {
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

all_strands = list(predictions.keys())

# ---------------------------------------------------------
# DIVIDER
# ---------------------------------------------------------
st.divider()

# ---------------------------------------------------------
# PREDICTION SECTION (TOP BOX)
# ---------------------------------------------------------
with st.container():
    st.subheader("🔮 Predict Next Year's Enrollment")

    strand = st.selectbox("Select Strand:", all_strands)

    if st.button("✨ Predict Next Year"):
        next_year = 2026
        predicted_value = predictions[strand]

        st.markdown(f"""
        <div style="padding: 15px; background-color: #E8F4FF; border-radius: 10px; border: 1px solid #BDD7EE;">
            <h3>🔮 Prediction for {strand} in {next_year}: 
            <span style="color:#005BBB;">{predicted_value} students</span></h3>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# DECISION TREE VISUALIZATION (AFTER PREDICTION, BEFORE HISTORICAL)
# ---------------------------------------------------------
if uploaded_file := st.file_uploader("📂 Upload Cleaned JRU SHS Dataset CSV File", type=["csv"]):
    st.success("📁 File loaded successfully!")

    df_tree = pd.read_csv(uploaded_file, parse_dates=["DateEnrolled", "Birthdate"])
    df_tree = df_tree[['YearLevel', 'Strand', 'Student']].dropna()

    X = df_tree[['YearLevel', 'Strand']]
    y = df_tree['Student']

    # Preprocessing for Decision Tree
    categorical_features = ['Strand']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )

    # Decision Tree pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(max_depth=10, random_state=42))
    ])

    model.fit(X, y)

    # Plot Decision Tree
    st.divider()
    with st.container():
        st.subheader("🌳 Decision Tree Visualization for Enrollment Prediction")

        tree_model = model.named_steps['regressor']
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()

        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(tree_model, filled=True, feature_names=feature_names, rounded=True, fontsize=10, ax=ax)
        plt.title("📈 Decision Tree for SHS Enrollment", fontsize=16)

        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------------------------------------
    # HISTORICAL VISUALIZATION (WHEN CSV IS UPLOADED)
    # ---------------------------------------------------------
    st.divider()
    st.subheader("📈 Historical Enrollment Dashboard")

    df_tree["Year"] = df_tree["DateEnrolled"].dt.year

    # Enrollment Count by Year
    st.write("### 🗓 Enrollment Count by Year")
    enroll_by_year = df_tree.groupby("Year").size().reset_index(name="Enrollment")
    st.line_chart(enroll_by_year.set_index("Year"))

    # Strand distribution
    st.write("### 🧭 Strand Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df_tree,
        x="Strand",
        palette="pastel",
        order=df_tree["Strand"].value_counts().index
    )
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)

    # Gender distribution
    st.write("### 🚹🚺 Gender Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    df_tree["Gender"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%"
    )
    plt.ylabel("")
    st.pyplot(fig)
    plt.close(fig)

    # Year vs Strand stacked chart
    st.write("### 📊 Enrollment by Year & Strand")
    year_strand = df_tree.groupby(["Year", "Strand"]).size().unstack(fill_value=0)
    st.bar_chart(year_strand)
