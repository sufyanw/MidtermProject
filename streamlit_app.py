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

df = pd.read_csv("student_data.csv")
app_page = st.sidebar.selectbox("Select Page", ['Introduction', 'Data Exploration', 'Visualization', 'Prediction'])

if app_page == 'Introduction':
    st.title("Student Alcohol Consumption App")
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

elif app_page == 'Data Exploration':
    
    st.title("Data Exploration")

    st.subheader("01 Description of the Dataset")

    st.dataframe(df.describe())

    st.subheader("02 Missing values")

    dfnull = df.isnull()/len(df)*100
    total_missing = dfnull.sum().round(2)
    st.write(total_missing)
    if total_missing[0] == 0.0:
        st.success("Congrats, there are no missing values!")
    else:
        st.error("There are missing values.")

    if st.button("Generate Report"):
        #function to load html file
        def read_html_report(file_path):
            with codecs.open(file_path, 'r', encoding="utf-8") as f:
                return f.read()
        
        html_report = read_html_report('report.html')
        
        #displaying file
        st.title("Streamlit Quality Report")
        
        st.components.v1.html(html_report, height=1000,scrolling=True)


elif app_page == 'Visualization':

    st.title("Data Visualization")

    list_columns = df.columns
    values = st.multiselect("Select 2 variables:", list_columns, ["Dalc","Walc"])

    #creation of the line chart
    st.line_chart(df, x=values[0], y=values[1])

    #creation of bar chart
    st.bar_chart(df, x=values[0], y=values[1])


    values_pairplot = st.multiselect("Select 4 variables:", list_columns, ["Dalc","Walc", "studytime", "absences"])

    df2 = df[[values_pairplot[0],values_pairplot[1],values_pairplot[2],values_pairplot[3]]]
    st.pyplot(sns.pairplot(df2))

    st.subheader("Distribution of Alcohol Consumption Among Students") 

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(df['Dalc'], bins=5, kde=False, ax=axes[0])
    axes[0].set_title('Weekday Alcohol Consumption (Dalc)')

    sns.histplot(df['Walc'], bins=5, kde=False, ax=axes[1])
    axes[1].set_title('Weekend Alcohol Consumption (Walc)')

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Correlation Heatmap of Numerical Variables")

    numerical_columns = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_columns].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, square=True, ax=ax)

    plt.title('Correlation Matrix of Numerical Variables')
    
    st.pyplot(fig)

    st.subheader("Jointplot of Weekday Alcohol Consumption (Dalc) vs Final Course Grade (G3)")

    plt.figure(figsize=(10, 6))
    sns.jointplot(data=df, x='Dalc', y='G3', kind='scatter')

    plt.suptitle('Relationship between Weekday Alcohol Consumption (Dalc) and Final Course Grade (G3)', y=1.03)
    plt.subplots_adjust(top=0.95)

    st.pyplot(plt)


elif app_page == "Prediction":
    st.title("Prediction")
    
    # Sample size selection
    sample_size = st.sidebar.slider("Select sample size from Dataset", min_value=10, max_value=100, step=10, value=20)
    df_sample = df.sample(frac=sample_size / 100)

    list_columns = df.columns
    input_lr = st.multiselect("Select variables:", list_columns, ["Dalc", "Walc"])

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
        st.write("Mean Absolute Error:", mae)
        st.write("R² Output:", r2)