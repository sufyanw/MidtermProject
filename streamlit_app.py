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

st.title("Student Alcohol Consumption App")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    image_path = Image.open("image.png")
    st.image(image_path, width=400)

st.write("""
## Introduction
This app is designed to explore and analyze student alcohol consumption data, with a specific focus on high school students in Portugal. The data used in this application is based on a study conducted on students from two Portuguese schools, one located in the city and the other in a rural area.

Alcohol consumption among students is a critical public health concern, as it can lead to various long-term effects on health, academic performance, and social well-being. This dataset provides valuable insights into the social, familial, and personal factors that may influence alcohol consumption habits in adolescents.

## Dataset Background
The dataset used in this app is derived from a survey conducted on secondary school students. It contains detailed information about students' alcohol consumption patterns on weekdays and weekends, along with other demographic, familial, and academic factors. The main goal is to analyze how different variables such as family background, social relationships, and academic performance affect students' alcohol consumption behavior.

## Objective
This app aims to:
- Explore student alcohol consumption patterns in Portugal.
- Identify the key factors influencing alcohol consumption.
- Provide insights into the impact of alcohol consumption on students' academic performance and social behavior.

Understanding the factors that contribute to alcohol consumption in students can help educators, parents, and policymakers create effective prevention and intervention strategies to mitigate its negative effects.

## Key Features
- Visualization of student alcohol consumption on both weekdays and weekends.
- Analysis of the relationships between alcohol consumption and various factors such as family structure, peer influence, and academic performance.
- Machine learning models to predict alcohol consumption based on input features.

Feel free to explore the data and uncover interesting patterns about student alcohol consumption in Portugal!
""")