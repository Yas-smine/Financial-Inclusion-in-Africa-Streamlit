import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("Financial Inclusion Prediction App")

# ----------- Load model and encoders ----------- 
model = joblib.load("bank_model.pkl")
encoders = joblib.load("label_encoder.pkl")   #Important

st.header("Enter Person Information")

# ----------- Functions to simplify features --------------
def simplify_education(x):
    if x == "No formal education":
        return "none"
    elif x == "Primary education":
        return "primary"
    elif x == "Secondary education":
        return "secondary"
    elif x in ["Vocational/Specialised training", "Tertiary education"]:
        return "higher"
    elif x == "Other/Dont know/RTA":
        return "other"
    else:
        return "other"  # valeur par défaut pour éviter les erreurs

    
def simplify_job(job):
    if job in ["Formally employed", "Government Dependent"]:
        return "formal"
    elif job in ["Informally employed", "Self employed", "Farming"]:
        return "informal"
    else:
        return "none"


# ----------- Inputs --------------
country = st.selectbox("Country", ["Kenya", "Uganda", "Tanzania", "Rwanda"])
year = st.selectbox("Year", [2016, 2017, 2018])
location = st.selectbox("Location Type", ["Urban", "Rural"])
cell = st.selectbox("Cellphone Access", ["Yes", "No"])
household_size = st.number_input("Household Size", min_value=1, max_value=30)
age = st.number_input("Age", min_value=15, max_value=100)
gender = st.selectbox("Gender", ["Male", "Female"])
relationship = st.selectbox(
    "Relationship with Head",
    ['Spouse', 'Head of Household', 'Other relative', 'Child', 'Parent',
       'Other non-relatives']
)
marital = st.selectbox(
    "Marital Status",
    ['Married/Living together', 'Widowed', 'Single/Never Married', 'Divorced/Seperated', 'Dont know']
)
education = st.selectbox(
    "Education Level",
    ['Secondary education', 'No formal education', 'Vocational/Specialised training', 
     'Primary education', 'Tertiary education', 'Other/Dont know/RTA'
    ]
)

job = st.selectbox(
    "Job Type",
    [
        "Self employed", "Formally employed Private","Formally employed Government", 
        "Informally employed","Farming and Fishing", "Remittance Dependent",
        "Government Dependent", "Other Income","No Income"
    ]
)



# ----------- Predict button --------------
if st.button("Predict"):
    
    input_data = {
        "country": country,
        "year": year,
        "location_type": location,
        "cellphone_access": cell,
        "household_size": household_size,
        "age_of_respondent": age,
        "gender_of_respondent": gender,
        "relationship_with_head": relationship,
        "marital_status": marital,
        "education_level": education,
        "job_type": job
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Simplify features
    input_df["education_level"] = input_df["education_level"].apply(simplify_education)
    input_df["job_type"] = input_df["job_type"].apply(simplify_job)


    # ----------- Apply LabelEncoders exactly like training ----------- 
    for col in input_df.columns:
        if col in encoders:
            le = encoders[col]
            input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            input_df[col] = encoders[col].transform(input_df[col])

    # ----------- Predict ----------- 
    prediction = model.predict(input_df)[0]

    # ----------- Display result ----------- 
    if prediction == 1:
        st.success("This person is likely to HAVE a bank account.")
    else:
        st.warning("This person is NOT likely to have a bank account.")
