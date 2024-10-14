#Student Alcohol Consumption App
import streamlit as st
import pandas as pd
import numpy as np
#for displaying images
from PIL import Image
import seaborn as sns
import codecs
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

st.set_page_config(page_title='Student Alcohol Consumption App')
df = pd.read_csv("student_data.csv")

selected = option_menu(
  menu_title = None,
  options = ["Introduction","Exploration","Visualization","Prediction"],
  icons=["book", "search", "bar-chart-line", "magic"],
  default_index = 0,
  orientation = "horizontal",
)

if selected == 'Introduction':
    st.title("Student Alcohol Consumption 🍷")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image_path = Image.open("image.png")
        st.image(image_path, width=400)

    st.write("""
    ## Introduction
    This app is designed to explore and analyze student alcohol consumption data, with a specific focus on high school students in Portugal.

    Alcohol consumption among students is a critical public health concern, as it can lead to various long-term effects on health, academic performance, and social well-being. This dataset provides valuable insights into the social, familial, and personal factors that may influence alcohol consumption habits in adolescents.

    ## Dataset Background
    The dataset used in this app is derived from a survey conducted on secondary school students. It contains detailed information about students' alcohol consumption patterns on weekdays and weekends, along with other demographic, familial, and academic factors. The main goal is to analyze how different variables such as family background, social relationships, and academic performance affect students' alcohol consumption behavior.

    ## Objective
    This app aims to:
    - Identify the key factors influencing alcohol consumption.
    - Provide insights into the impact of alcohol consumption on students' academic performance and social behavior.

    Understanding the factors that contribute to alcohol consumption in students can help educators, parents, and policymakers create effective prevention.

    ## Key Features
    - Visualization of student alcohol consumption on both weekdays & weekends.
    - Analysis of the relationships between alcohol consumption and various factors such as family structure, peer influence, and academic performance.

    """)

elif selected == 'Exploration':
    
    st.title("Data Exploration 🔍")
    
    # Create tabs for different sections of data exploration
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dataset Head", "Dataset Tail", "Description", "Missing Values", "Generate Report"])

    with tab1:
        st.subheader("Head of the Dataset")
        st.dataframe(df.head())

    with tab2:
        st.subheader("Tail of the Dataset")
        st.dataframe(df.tail())

    with tab3:
        st.subheader("Description of the Dataset")
        st.dataframe(df.describe())

    with tab4:
        st.subheader("Missing values")
        dfnull = df.isnull()/len(df)*100
        total_missing = dfnull.sum().round(2)
        st.write(total_missing)
        if total_missing[0] == 0.0:
            st.success("Congrats, there are no missing values!")
        else:
            st.error("There are missing values.")

    with tab5:
        if st.button("Generate Report"):
            # Function to load the HTML report
            def read_html_report(file_path):
                with codecs.open(file_path, 'r', encoding="utf-8") as f:
                    return f.read()
            
            html_report = read_html_report('report.html')
            
            # Displaying the HTML report
            st.title("Streamlit Quality Report")
            st.components.v1.html(html_report, height=1000, scrolling=True)



elif selected == 'Visualization':
    st.title("Data Visualization 📈")

    tab1, tab2, tab3, tab4 = st.tabs(["Line & Bar Charts", "Pairplot", "Distribution", "Correlation Heatmap"])

    with tab1:
        st.subheader("Line & Bar Charts")
        list_columns = df.columns
        values = st.multiselect("Select 2 variables:", list_columns, ["Dalc", "studytime"])

        if len(values) == 2:
            st.line_chart(df.set_index(values[0])[values[1]])
            st.bar_chart(df.set_index(values[0])[values[1]])
        else:
            st.warning("Please select exactly 2 variables.")

    with tab2:
        st.subheader("Pairplot")
        
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        values_pairplot = st.multiselect("Select 4 variables:", numeric_columns, ["Dalc", "Walc", "studytime", "absences"])

        if len(values_pairplot) == 4:
            df2 = df[values_pairplot]
            st.pyplot(sns.pairplot(df2))
        else:
            st.warning("Please select exactly 4 variables.")

    with tab3:
        st.subheader("Distribution of Alcohol Consumption Among Students")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.histplot(df['Dalc'], bins=5, kde=False, ax=axes[0])
        axes[0].set_title('Weekday Alcohol Consumption (Dalc)')

        sns.histplot(df['Walc'], bins=5, kde=False, ax=axes[1])
        axes[1].set_title('Weekend Alcohol Consumption (Walc)')

        plt.tight_layout()
        st.pyplot(fig)
        st.write("""
### Analysis of Alcohol Consumption Distributions

1. **Weekday Alcohol Consumption (Dalc)**:
   - The majority of students report very low alcohol consumption during weekdays, with the highest counts in the **1.0** category. 
   - This indicates that students generally minimize alcohol intake due to academic & familial responsibilities.

2. **Weekend Alcohol Consumption (Walc)**:
   - Weekend consumption shows a different trend, with higher frequencies in the **1.0** and **2.0** categories, and a gradual increase into moderate levels (3.0 and above).
   - Suggests students feel more inclined to consume alcohol socially during weekends.
""")


    with tab4:
        st.subheader("Correlation Heatmap of Numerical Variables")
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numerical_columns].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True, ax=ax)

        plt.title('Correlation Matrix of Numerical Variables')
        st.pyplot(fig)

        st.write("""
### Correlation Heatmap Analysis

The heatmap illustrates the relationships between various factors influencing student behavior and performance. Notable correlations include:

- **Alcohol Consumption:** 
  - Weekend alcohol consumption (Walc) has a strong positive correlation (0.65) with weekday consumption (Dalc), indicating that students who drink more on weekends also tend to drink more during the week. If action is not taken for extreme cases, then the excessive alcohol intake can negatively impact school performance and familial relationships.
  
- **Academic Performance:**
  - Final grades (G3) show a positive correlation with study time (0.43) and family relationships (0.27), suggesting that better study habits and family support may contribute to higher academic achievement.

- **Social Factors:**
  - Freetime (0.29) and going out with friends (goout) (0.21) positively correlate with final grades (G3), indicating that students who engage socially and have more leisure time may perform better academically.
""")




elif selected == "Prediction":
    st.title("Prediction 🔮")
    
    # Sample size selection
    sample_size = st.sidebar.slider("Select sample size from Dataset", min_value=10, max_value=100, step=10, value=20)
    df_sample = df.sample(frac=sample_size / 100)

    list_columns = df.columns
    input_lr = st.multiselect("Select variables:", list_columns, ["Walc", "studytime"])

    # Check if at least one variable is selected
    if input_lr:
        df2 = df_sample[input_lr]
        y = df_sample["Walc"] 
   
        test_size = st.sidebar.slider("Select test size (in percentage)", min_value=10, max_value=90, value=20)
        test_size = test_size / 100 

        # Step 1: Split into X and y
        X = df2

        # Step 2: Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # Step 3: Initialize the linear regression model
        lr = LinearRegression()

        # Step 4: Train the model
        lr.fit(X_train, y_train)

        # Step 5: Prediction
        predictions = lr.predict(X_test)

        # Step 6: Evaluate
        mae = metrics.mean_absolute_error(predictions, y_test)
        r2 = metrics.r2_score(predictions, y_test)
        mae_round = round(mae, 2)
        r2_round = round(r2, 2)
        st.write("Mean Absolute Error:", mae_round)
        st.write("R² Output:", r2_round)
    else:
        st.warning("Please select at least one value.")