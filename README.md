# 🍷 Student Alcohol Consumption App

## Overview
The **Student Alcohol Consumption App** is a Streamlit-based web application that allows users to explore and analyze alcohol consumption patterns among high school students in Portugal. Using machine learning models and data visualizations, the app highlights key factors that influence student alcohol consumption and its potential impact on academic performance and social behavior.

## Features
- **Interactive Data Visualizations:** Visualize alcohol consumption patterns on weekdays and weekends.
- **Data Insights:** Analyze relationships between various demographic, familial, and academic factors and student alcohol consumption.
- **Machine Learning Predictions:** Use a linear regression model to predict alcohol consumption based on selected input features.

## Dataset
The dataset used in this app comes from a survey conducted on secondary school students from two Portuguese schools: one urban and one rural. It includes details on student alcohol consumption and several other factors:

- school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
- sex - student's sex (binary: 'F' - female or 'M' - male)
- age - student's age (numeric: from 15 to 22)
- address - student's home address type (binary: 'U' - urban or 'R' - rural)
- famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
- Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
- Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
- Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
- Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
- Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
- reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
- guardian - student's guardian (nominal: 'mother', 'father' or 'other')
- traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
- studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
- failures - number of past class failures (numeric: n if 1<=n<3, else 4)
- schoolsup - extra educational support (binary: yes or no)
- famsup - family educational support (binary: yes or no)
paid - extra paid classes within the course subject (Portuguese) (binary: yes or no)
- activities - extra-curricular activities (binary: yes or no)
- nursery - attended nursery school (binary: yes or no)
higher - wants to take higher education (binary: yes or no)
- internet - Internet access at home (binary: yes or no)
- romantic - with a romantic relationship (binary: yes or no)
famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
- freetime - free time after school (numeric: from 1 - very low to 5 - very high)
- goout - going out with friends (numeric: from 1 - very low to 5 - very high)
- Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
- Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
- health - current health status (numeric: from 1 - very bad to 5 - very good)
- absences - number of school absences (numeric: from 0 to 93)

These grades are related with the Portugese course:
- G1 - first period grade (numeric: from 0 to 20)
- G2 - second period grade (numeric: from 0 to 20)
- G3 - final grade (numeric: from 0 to 20, output target)

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