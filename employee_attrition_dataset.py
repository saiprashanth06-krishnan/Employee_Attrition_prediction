import streamlit as st
import pickle
import numpy as np

# Apply background color using custom CSS
st.markdown(
    """
    <style>
    body {
        background-color: #ADD8E6; /* Sky blue */
    }
    .stApp {
        background-color: #ADD8E6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define model file paths
model_paths = {
    "Employee Attrition Prediction": "C:/Users/User/Downloads/Employee_attrition_Attrition.pkl",
    "Performance Rating Prediction": "C:/Users/User/Downloads/Employee_attrition_performancerating.pkl"
}

# Define features for each model
feature_dict = {
    "Employee Attrition Prediction": [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 
        'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
        'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 
        'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely', 
        'Department_Research & Development', 'Department_Sales', 'EducationField_Life Sciences', 'EducationField_Marketing', 
        'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree', 'JobRole_Human Resources', 
        'JobRole_Laboratory Technician', 'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director', 
        'JobRole_Research Scientist', 'JobRole_Sales Executive', 'JobRole_Sales Representative', 'MaritalStatus_Married', 'MaritalStatus_Single'
    ],
    "Performance Rating Prediction": [
        'Education', 'JobInvolvement', 'JobLevel', 'MonthlyIncome', 'YearsAtCompany', 'YearsInCurrentRole'
    ]
}

# Sidebar for model selection
st.sidebar.title("Select a Model")
selected_model = st.sidebar.radio("Choose a prediction model:", list(model_paths.keys()))

# Load the selected model
with open(model_paths[selected_model], 'rb') as file:
    model = pickle.load(file)

# Display the selected model name
st.title(selected_model)

# Get corresponding features
features = feature_dict[selected_model]

# User input fields
user_input = []
st.subheader("Enter the following details:")
for feature in features:
    value = st.number_input(f"{feature}:", value=0.0)
    user_input.append(value)

# Prediction button
if st.button("Predict"):
    if selected_model == "Performance Rating Prediction":
        prediction = model.predict([np.array(user_input)])[0]
        probability = model.predict_proba([np.array(user_input)])[0][1]  # Probability-based prediction
        threshold = 0.50  # Set threshold for classification
        predicted_rating = 4 if probability >= threshold else 3
        st.success(f"Predicted Performance Rating: {predicted_rating}")
    else:
        # Get prediction probability for attrition model
        probability = model.predict_proba([np.array(user_input)])[0][1]  # Probability of attrition
        threshold = 0.23
        if probability <= threshold:
            st.success(f"Prediction: Yes - Employee may leave the company ")
        else:
            st.success(f"Prediction: No - Employee is likely to stay ")
