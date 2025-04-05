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
    "Employee Attrition Prediction": "C:/Users/User/Downloads/Employee_attrition_Attrition_1.pkl",
    "Performance Rating Prediction": "C:/Users/User/Downloads/Employee_attrition_performancerating.pkl"
}

# Define features for each model
feature_dict = {
    "Employee Attrition Prediction": [
        'Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
        'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
        'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked', 'OverTime',
        'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear'
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

# If selected model is "Employee Attrition Prediction", load encoders
label_encoders = None
if selected_model == "Employee Attrition Prediction":
    with open("C:/Users/User/Downloads/Employee_attrition_label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("C:/Users/User/Downloads/Employee_attrition_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)    

# Display the selected model name
st.title(selected_model)

# Get corresponding features
features = feature_dict[selected_model]

# Collect user input
user_input = []
st.subheader("Enter the following details:")

for feature in features:
    if selected_model == "Employee Attrition Prediction" and feature in ['JobRole', 'MaritalStatus', 'OverTime']:
        # Dropdown values from label encoders
        classes = label_encoders[feature].classes_
        selected = st.selectbox(f"{feature}:", list(classes))
        encoded = label_encoders[feature].transform([selected])[0]
        user_input.append(encoded)
    else:
        val = st.number_input(f"{feature}:", value=0.0)
        user_input.append(val)

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
        user_input_np = np.array(user_input).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input_np)
        probability = model.predict_proba(user_input_scaled)[0][1]
        threshold = 0.03
        if probability <= threshold:
            st.success(f"✅ Prediction: Yes - Employee may leave the company.")
        else:
            st.success(f"🔒 Prediction: No - Employee is likely to stay.")
