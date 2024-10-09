# 🍷 Student Alcohol Consumption App

## Overview
The **Student Alcohol Consumption App** is a Streamlit-based web application that allows users to explore and analyze alcohol consumption patterns among high school students in Portugal. Using machine learning models and data visualizations, the app highlights key factors that influence student alcohol consumption and its potential impact on academic performance and social behavior.

## Features
- **Interactive Data Visualizations:** Visualize alcohol consumption patterns on weekdays and weekends.
- **Data Insights:** Analyze relationships between various demographic, familial, and academic factors and student alcohol consumption.
- **Machine Learning Predictions:** Use a linear regression model to predict alcohol consumption based on selected input features.

## Dataset
The dataset used in this app comes from a survey conducted on secondary school students from two Portuguese schools: one urban and one rural. It includes details on student alcohol consumption and several other factors:
- Demographic information (age, gender)
- Family background (parents’ education, family size)
- Social relationships (friendships, time spent with family)
- Academic performance

The primary objective is to understand how these factors influence alcohol consumption behavior in students.

## Installation and Usage

### Prerequisites
To run this project locally, ensure you have the following installed:
- Python 3.x
- Streamlit
- Pandas
- Numpy
- PIL (Python Imaging Library)
- Seaborn
- Scikit-learn

### Setup Instructions
1. Clone this repository to your local machine.
2. Install the required dependencies using pip:
```bash
pip install streamlit pandas numpy seaborn scikit-learn pillow